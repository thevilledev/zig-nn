package lab

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"nnctl/internal/process"
	"nnctl/internal/zig"
)

const (
	maxRunEvents     = 512
	maxRunEventBytes = 32 * 1024 * 1024
	maxCompletedRuns = 32
)

var (
	ErrBusy        = errors.New("an experiment is already running")
	ErrRunNotFound = errors.New("run not found")
)

type Event struct {
	Version    int             `json:"v"`
	RunID      string          `json:"run_id"`
	Sequence   uint64          `json:"seq"`
	Type       string          `json:"type"`
	Experiment string          `json:"experiment"`
	Step       *int            `json:"step,omitempty"`
	TotalSteps *int            `json:"total_steps,omitempty"`
	Data       json.RawMessage `json:"data,omitempty"`
}

type eventInput struct {
	Version    int             `json:"v"`
	Type       string          `json:"type"`
	Experiment string          `json:"experiment"`
	Step       *int            `json:"step,omitempty"`
	TotalSteps *int            `json:"total_steps,omitempty"`
	Data       json.RawMessage `json:"data,omitempty"`
}

type Executor interface {
	Execute(context.Context, ExperimentSpec, RunOptions, func([]byte) error, func(string)) error
}

type ExecutorFunc func(context.Context, ExperimentSpec, RunOptions, func([]byte) error, func(string)) error

func (f ExecutorFunc) Execute(ctx context.Context, spec ExperimentSpec, options RunOptions, stdout func([]byte) error, stderr func(string)) error {
	return f(ctx, spec, options, stdout, stderr)
}

type CommandExecutor struct {
	RepoRoot string
	Zig      string
	Mode     string
}

func (e CommandExecutor) Execute(ctx context.Context, spec ExperimentSpec, options RunOptions, stdoutLine func([]byte) error, stderrLine func(string)) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	mode := e.Mode
	if spec.Optimize != "" {
		mode = spec.Optimize
	}
	gpu := ""
	if options.Backend != "cpu" {
		gpu = options.Backend
	}
	args := zig.RunArgs(spec.Step, zig.Options{Optimize: mode, GPU: gpu}, options.Arguments)
	cmd := process.CommandContext(ctx, e.Zig, args...)
	cmd.Dir = e.RepoRoot
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("capture experiment stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("capture experiment stderr: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start experiment: %w", err)
	}

	var wg sync.WaitGroup
	scanErrors := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		scanErrors <- scanLines(stdout, func(line []byte) error {
			if err := stdoutLine(line); err != nil {
				cancel()
				return err
			}
			return nil
		})
	}()
	go func() {
		defer wg.Done()
		scanErrors <- scanLines(stderr, func(line []byte) error {
			stderrLine(string(line))
			return nil
		})
	}()

	wg.Wait()
	waitErr := cmd.Wait()
	close(scanErrors)
	for scanErr := range scanErrors {
		if scanErr != nil {
			return scanErr
		}
	}
	if waitErr != nil {
		return fmt.Errorf("experiment process: %w", waitErr)
	}
	return nil
}

func scanLines(reader io.Reader, visit func([]byte) error) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := append([]byte(nil), scanner.Bytes()...)
		if err := visit(line); err != nil {
			return err
		}
	}
	return scanner.Err()
}

type run struct {
	id         string
	experiment string
	cancel     context.CancelFunc
	done       chan struct{}

	mu          sync.Mutex
	events      []Event
	eventBytes  int
	nextSeq     uint64
	subscribers map[chan Event]struct{}
	completed   bool
	finished    bool
	nativeStart bool
	nativeStep  int
	nativeTotal int
}

type Manager struct {
	executor Executor
	ctx      context.Context
	cancel   context.CancelFunc

	mu        sync.Mutex
	runs      map[string]*run
	completed []string
	active    *run
	closed    bool
}

func NewManager(parent context.Context, executor Executor) *Manager {
	ctx, cancel := context.WithCancel(parent)
	return &Manager{executor: executor, ctx: ctx, cancel: cancel, runs: make(map[string]*run)}
}

func (m *Manager) Start(spec ExperimentSpec, options RunOptions) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", errors.New("run manager is closed")
	}
	if m.active != nil {
		return "", ErrBusy
	}

	ctx, cancel := context.WithCancel(m.ctx)
	r := &run{
		id:          newRunID(),
		experiment:  spec.ID,
		cancel:      cancel,
		done:        make(chan struct{}),
		subscribers: make(map[chan Event]struct{}),
	}
	m.runs[r.id] = r
	m.active = r
	go m.execute(ctx, r, spec, options)
	return r.id, nil
}

func (m *Manager) execute(ctx context.Context, r *run, spec ExperimentSpec, options RunOptions) {
	err := m.executor.Execute(ctx, spec, options, func(line []byte) error {
		input, parseErr := parseEvent(line, spec.ID)
		if parseErr != nil {
			return parseErr
		}
		if protocolErr := r.validateProtocol(input); protocolErr != nil {
			return protocolErr
		}
		r.append(input)
		return nil
	}, func(line string) {
		if line == "" {
			return
		}
		if message, ok := strings.CutPrefix(line, "cloud: "); ok {
			r.append(eventInput{
				Version:    1,
				Type:       "run_status",
				Experiment: spec.ID,
				Data:       marshalData(map[string]string{"message": message}),
			})
		}
		r.append(eventInput{
			Version:    1,
			Type:       "log",
			Experiment: spec.ID,
			Data:       marshalData(map[string]string{"message": line}),
		})
	})

	if err != nil {
		message := err.Error()
		cancelled := errors.Is(ctx.Err(), context.Canceled)
		if cancelled {
			message = "run cancelled"
		}
		r.append(eventInput{
			Version:    1,
			Type:       "run_failed",
			Experiment: spec.ID,
			Data: marshalData(map[string]any{
				"message":   message,
				"cancelled": cancelled,
			}),
		})
	} else if !r.hasCompleted() {
		r.append(eventInput{
			Version:    1,
			Type:       "run_failed",
			Experiment: spec.ID,
			Data:       marshalData(map[string]string{"message": "experiment exited without a completion event"}),
		})
	}
	m.mu.Lock()
	if m.active == r {
		m.active = nil
	}
	m.completed = append(m.completed, r.id)
	for len(m.completed) > maxCompletedRuns {
		oldest := m.completed[0]
		m.completed = m.completed[1:]
		delete(m.runs, oldest)
	}
	m.mu.Unlock()
	r.finish()
}

func (r *run) validateProtocol(input eventInput) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch input.Type {
	case "run_started":
		if r.nativeStart {
			return errors.New("experiment emitted run_started more than once")
		}
		if r.completed {
			return errors.New("experiment emitted run_started after completion")
		}
		r.nativeStart = true
		return nil
	case "metric", "snapshot", "run_completed":
		if !r.nativeStart {
			return fmt.Errorf("experiment emitted %s before run_started", input.Type)
		}
		if r.completed {
			return fmt.Errorf("experiment emitted %s after run_completed", input.Type)
		}
		if input.Step == nil || input.TotalSteps == nil {
			return fmt.Errorf("event %s must include step and total_steps", input.Type)
		}
		if *input.TotalSteps <= 0 || *input.Step < 0 || *input.Step > *input.TotalSteps {
			return fmt.Errorf("event %s has invalid step %d of %d", input.Type, *input.Step, *input.TotalSteps)
		}
		if r.nativeTotal != 0 && r.nativeTotal != *input.TotalSteps {
			return fmt.Errorf("event %s changed total_steps from %d to %d", input.Type, r.nativeTotal, *input.TotalSteps)
		}
		if *input.Step < r.nativeStep {
			return fmt.Errorf("event %s moved backwards from step %d to %d", input.Type, r.nativeStep, *input.Step)
		}
		if input.Type == "run_completed" && *input.Step != *input.TotalSteps {
			return fmt.Errorf("run_completed must report the final step")
		}
		r.nativeStep = *input.Step
		r.nativeTotal = *input.TotalSteps
		return nil
	default:
		return fmt.Errorf("unsupported native event type %q", input.Type)
	}
}

func parseEvent(line []byte, experiment string) (eventInput, error) {
	var input eventInput
	if err := json.Unmarshal(line, &input); err != nil {
		return eventInput{}, fmt.Errorf("decode experiment event: %w", err)
	}
	if input.Version != 1 {
		return eventInput{}, fmt.Errorf("unsupported experiment event version %d", input.Version)
	}
	if input.Experiment != experiment {
		return eventInput{}, fmt.Errorf("event experiment %q does not match %q", input.Experiment, experiment)
	}
	switch input.Type {
	case "run_started", "metric", "snapshot", "run_completed":
	default:
		return eventInput{}, fmt.Errorf("unsupported experiment event type %q", input.Type)
	}
	if len(input.Data) == 0 || !json.Valid(input.Data) {
		return eventInput{}, fmt.Errorf("event %s has invalid data", input.Type)
	}
	return input, nil
}

func (r *run) append(input eventInput) Event {
	r.mu.Lock()
	r.nextSeq++
	event := Event{
		Version:    input.Version,
		RunID:      r.id,
		Sequence:   r.nextSeq,
		Type:       input.Type,
		Experiment: input.Experiment,
		Step:       input.Step,
		TotalSteps: input.TotalSteps,
		Data:       input.Data,
	}
	r.events = append(r.events, event)
	r.eventBytes += eventRetainedBytes(event)
	for len(r.events) > maxRunEvents || r.eventBytes > maxRunEventBytes {
		r.eventBytes -= eventRetainedBytes(r.events[0])
		copy(r.events, r.events[1:])
		r.events = r.events[:len(r.events)-1]
	}
	if event.Type == "run_completed" {
		r.completed = true
	}
	subscribers := make([]chan Event, 0, len(r.subscribers))
	for subscriber := range r.subscribers {
		subscribers = append(subscribers, subscriber)
	}
	r.mu.Unlock()

	for _, subscriber := range subscribers {
		select {
		case subscriber <- event:
		default:
		}
	}
	return event
}

func eventRetainedBytes(event Event) int {
	return len(event.Type) + len(event.Experiment) + len(event.Data) + 64
}

func (r *run) hasCompleted() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.completed
}

func (r *run) finish() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return
	}
	r.finished = true
	close(r.done)
}

type Subscription struct {
	Replay []Event
	Events <-chan Event
	Done   <-chan struct{}
	close  func()
}

func (s Subscription) Close() {
	if s.close != nil {
		s.close()
	}
}

func (m *Manager) Subscribe(id string, after uint64) (Subscription, error) {
	r, err := m.lookup(id)
	if err != nil {
		return Subscription{}, err
	}
	channel := make(chan Event, 256)
	r.mu.Lock()
	replay := eventsAfter(r.events, after)
	r.subscribers[channel] = struct{}{}
	r.mu.Unlock()
	return Subscription{
		Replay: replay,
		Events: channel,
		Done:   r.done,
		close: func() {
			r.mu.Lock()
			delete(r.subscribers, channel)
			r.mu.Unlock()
		},
	}, nil
}

func (m *Manager) EventsSince(id string, after uint64) ([]Event, error) {
	r, err := m.lookup(id)
	if err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return eventsAfter(r.events, after), nil
}

func eventsAfter(events []Event, after uint64) []Event {
	result := make([]Event, 0, len(events))
	for _, event := range events {
		if event.Sequence > after {
			result = append(result, event)
		}
	}
	return result
}

func (m *Manager) Cancel(id string) error {
	r, err := m.lookup(id)
	if err != nil {
		return err
	}
	r.cancel()
	return nil
}

func (m *Manager) Close() {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return
	}
	m.closed = true
	active := m.active
	m.cancel()
	m.mu.Unlock()
	if active != nil {
		<-active.done
	}
}

func (m *Manager) lookup(id string) (*run, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	r, ok := m.runs[id]
	if !ok {
		return nil, ErrRunNotFound
	}
	return r, nil
}

func newRunID() string {
	var bytes [8]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		panic(fmt.Sprintf("generate run id: %v", err))
	}
	return hex.EncodeToString(bytes[:])
}

func marshalData(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("marshal run event: %v", err))
	}
	return encoded
}
