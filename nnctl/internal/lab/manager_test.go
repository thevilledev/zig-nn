package lab

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestManagerOrdersAndReplaysEvents(t *testing.T) {
	executor := ExecutorFunc(func(_ context.Context, spec ExperimentSpec, _ RunOptions, stdout func([]byte) error, stderr func(string)) error {
		stderr("building experiment")
		for _, event := range []map[string]any{
			{"v": 1, "type": "run_started", "experiment": spec.ID, "data": map[string]any{"config": map[string]any{}}},
			{"v": 1, "type": "metric", "experiment": spec.ID, "step": 1, "total_steps": 1, "data": map[string]any{"name": "loss", "value": 0.5}},
			{"v": 1, "type": "run_completed", "experiment": spec.ID, "step": 1, "total_steps": 1, "data": map[string]any{"final_loss": 0.5}},
		} {
			encoded, _ := json.Marshal(event)
			if err := stdout(encoded); err != nil {
				return err
			}
		}
		return nil
	})
	manager := NewManager(t.Context(), executor)
	defer manager.Close()
	spec, _ := ResolveExperiment("xor-training")
	id, err := manager.Start(spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	subscription, err := manager.Subscribe(id, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Close()
	select {
	case <-subscription.Done:
	case <-time.After(2 * time.Second):
		t.Fatal("run did not finish")
	}
	events, err := manager.EventsSince(id, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 4 {
		t.Fatalf("got %d events: %#v", len(events), events)
	}
	for index, event := range events {
		if event.Sequence != uint64(index+1) {
			t.Fatalf("event %d has sequence %d", index, event.Sequence)
		}
		if event.RunID != id {
			t.Fatalf("event run id %q != %q", event.RunID, id)
		}
	}
	if events[0].Type != "log" || events[len(events)-1].Type != "run_completed" {
		t.Fatalf("unexpected event types: %#v", events)
	}
	remaining, err := manager.EventsSince(id, 2)
	if err != nil || len(remaining) != 2 {
		t.Fatalf("replay after sequence 2 = %#v, %v", remaining, err)
	}
}

func TestManagerRejectsConcurrentRunAndCancels(t *testing.T) {
	started := make(chan struct{})
	executor := ExecutorFunc(func(ctx context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	})
	manager := NewManager(t.Context(), executor)
	spec, _ := ResolveExperiment("xor-training")
	id, err := manager.Start(spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	if _, err := manager.Start(spec, RunOptions{}); !errors.Is(err, ErrBusy) {
		t.Fatalf("second Start error = %v", err)
	}
	if err := manager.Cancel(id); err != nil {
		t.Fatal(err)
	}
	subscription, err := manager.Subscribe(id, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer subscription.Close()
	select {
	case <-subscription.Done:
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled run did not finish")
	}
	events, _ := manager.EventsSince(id, 0)
	if len(events) != 1 || events[0].Type != "run_failed" || !strings.Contains(string(events[0].Data), `"cancelled":true`) {
		t.Fatalf("cancel events = %#v", events)
	}
	manager.Close()
}

func TestManagerTurnsMalformedOutputIntoFailure(t *testing.T) {
	manager := NewManager(t.Context(), ExecutorFunc(func(_ context.Context, _ ExperimentSpec, _ RunOptions, stdout func([]byte) error, _ func(string)) error {
		return stdout([]byte("not-json"))
	}))
	defer manager.Close()
	spec, _ := ResolveExperiment("regression")
	id, err := manager.Start(spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	subscription, _ := manager.Subscribe(id, 0)
	defer subscription.Close()
	<-subscription.Done
	events, _ := manager.EventsSince(id, 0)
	if len(events) != 1 || events[0].Type != "run_failed" || !strings.Contains(string(events[0].Data), "decode experiment event") {
		t.Fatalf("malformed-output events = %#v", events)
	}
}

func TestManagerReportsChildFailure(t *testing.T) {
	manager := NewManager(t.Context(), ExecutorFunc(func(_ context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
		return errors.New("native process failed")
	}))
	defer manager.Close()
	spec, _ := ResolveExperiment("regression")
	id, err := manager.Start(spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	subscription, _ := manager.Subscribe(id, 0)
	defer subscription.Close()
	<-subscription.Done
	events, _ := manager.EventsSince(id, 0)
	if len(events) != 1 || events[0].Type != "run_failed" || !strings.Contains(string(events[0].Data), "native process failed") {
		t.Fatalf("child-failure events = %#v", events)
	}
}

func TestManagerRejectsOutOfOrderProtocol(t *testing.T) {
	manager := NewManager(t.Context(), ExecutorFunc(func(_ context.Context, spec ExperimentSpec, _ RunOptions, stdout func([]byte) error, _ func(string)) error {
		line, _ := json.Marshal(map[string]any{
			"v": 1, "type": "metric", "experiment": spec.ID,
			"step": 1, "total_steps": 10, "data": map[string]any{"name": "loss", "value": 1},
		})
		return stdout(line)
	}))
	defer manager.Close()
	spec, _ := ResolveExperiment("xor-training")
	id, err := manager.Start(spec, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	subscription, _ := manager.Subscribe(id, 0)
	defer subscription.Close()
	<-subscription.Done
	events, _ := manager.EventsSince(id, 0)
	if len(events) != 1 || events[0].Type != "run_failed" || !strings.Contains(string(events[0].Data), "before run_started") {
		t.Fatalf("out-of-order events = %#v", events)
	}
}

func TestManagerCloseCancelsActiveExecutor(t *testing.T) {
	stopped := make(chan struct{})
	manager := NewManager(t.Context(), ExecutorFunc(func(ctx context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
		<-ctx.Done()
		close(stopped)
		return ctx.Err()
	}))
	spec, _ := ResolveExperiment("xor-training")
	if _, err := manager.Start(spec, RunOptions{}); err != nil {
		t.Fatal(err)
	}
	manager.Close()
	select {
	case <-stopped:
	default:
		t.Fatal("executor was not stopped before manager close returned")
	}
	if _, err := manager.Start(spec, RunOptions{}); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("Start() after Close() error = %v", err)
	}
}

func TestManagerBoundsRetainedEvents(t *testing.T) {
	t.Parallel()
	r := &run{id: "run", experiment: "experiment", subscribers: make(map[chan Event]struct{})}
	payload := json.RawMessage(`{"message":"` + strings.Repeat("x", 1024*1024) + `"}`)
	for range 40 {
		r.append(eventInput{Version: 1, Type: "log", Experiment: r.experiment, Data: payload})
	}
	if len(r.events) >= 40 {
		t.Fatalf("retained %d events, want fewer than 40", len(r.events))
	}
	if r.eventBytes > maxRunEventBytes {
		t.Fatalf("retained %d bytes, limit is %d", r.eventBytes, maxRunEventBytes)
	}
}

func TestManagerEvictsOldCompletedRuns(t *testing.T) {
	t.Parallel()
	manager := NewManager(t.Context(), ExecutorFunc(func(_ context.Context, _ ExperimentSpec, _ RunOptions, _ func([]byte) error, _ func(string)) error {
		return errors.New("finished")
	}))
	defer manager.Close()
	spec, _ := ResolveExperiment("regression")
	ids := make([]string, 0, maxCompletedRuns+1)
	for range maxCompletedRuns + 1 {
		id, err := manager.Start(spec, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}
		subscription, err := manager.Subscribe(id, 0)
		if err != nil {
			t.Fatal(err)
		}
		<-subscription.Done
		subscription.Close()
		ids = append(ids, id)
	}
	if _, err := manager.EventsSince(ids[0], 0); !errors.Is(err, ErrRunNotFound) {
		t.Fatalf("oldest run lookup error = %v, want %v", err, ErrRunNotFound)
	}
	if _, err := manager.EventsSince(ids[len(ids)-1], 0); err != nil {
		t.Fatalf("newest run lookup error = %v", err)
	}
}
