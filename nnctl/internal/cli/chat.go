package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"strconv"
	"time"

	"nnctl/internal/zig"
)

type chatOptions struct {
	mode             string
	host             string
	port             int
	apiHost          string
	apiPort          int
	modelPath        string
	modelName        string
	maxTokens        int
	temperature      float64
	topK             int
	allowUntrained   bool
	readyTimeoutText string
}

func defaultChatOptions() chatOptions {
	return chatOptions{
		mode:             getenvDefault("NNCTL_OPTIMIZE", getenvDefault("BUILD_MODE", defaultBuildMode)),
		host:             "127.0.0.1",
		port:             8090,
		apiHost:          "127.0.0.1",
		apiPort:          8080,
		modelName:        "tiny-gpt-zig",
		maxTokens:        120,
		temperature:      1.0,
		topK:             8,
		readyTimeoutText: "2m",
	}
}

func (a *app) runChat(ctx context.Context, opts chatOptions) error {
	readyTimeout, err := validateChatOptions(opts)
	if err != nil {
		return err
	}

	apiBaseURL := "http://" + net.JoinHostPort(opts.apiHost, strconv.Itoa(opts.apiPort))
	serverArgs := []string{
		"--host", opts.apiHost,
		"--port", strconv.Itoa(opts.apiPort),
		"--model-name", opts.modelName,
		"--max-tokens", strconv.Itoa(opts.maxTokens),
		"--temperature", strconv.FormatFloat(opts.temperature, 'f', -1, 64),
		"--top-k", strconv.Itoa(opts.topK),
	}
	if opts.modelPath != "" {
		serverArgs = append(serverArgs, "--model", opts.modelPath)
	} else {
		serverArgs = append(serverArgs, "--allow-untrained")
	}

	runArgs := zig.RunArgs("run_tiny_gpt_openai", zig.Options{Optimize: opts.mode}, serverArgs)
	if _, err := fmt.Fprintf(a.stderr(), "==> %s\n", zig.CommandString(a.zig, runArgs)); err != nil {
		return fmt.Errorf("write inference command: %w", err)
	}
	serverCmd := exec.CommandContext(ctx, a.zig, runArgs...)
	serverCmd.Dir = a.repoRoot
	serverCmd.Stdin = a.stdin()
	serverCmd.Stdout = a.stderr()
	serverCmd.Stderr = a.stderr()
	if err := serverCmd.Start(); err != nil {
		return fmt.Errorf("start inference server: %w", err)
	}

	childDone := make(chan error, 1)
	go func() {
		childDone <- serverCmd.Wait()
	}()

	childConsumed, err := waitForInference(ctx, apiBaseURL+"/v1/models", readyTimeout, childDone)
	if err != nil {
		if !childConsumed {
			_ = serverCmd.Process.Kill()
			<-childDone
		}
		return err
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", serveChatApp)
	mux.Handle("/api/chat", chatProxy{
		apiBaseURL:  apiBaseURL,
		modelName:   opts.modelName,
		maxTokens:   opts.maxTokens,
		temperature: opts.temperature,
		client:      &http.Client{Timeout: 2 * time.Minute},
	})

	chatAddr := net.JoinHostPort(opts.host, strconv.Itoa(opts.port))
	chatServer := &http.Server{
		Addr:              chatAddr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	httpDone := make(chan error, 1)
	output := newErrorWriter(a.stdout())
	output.printf("chat app: http://%s\n", chatAddr)
	output.printf("inference: %s\n", apiBaseURL)
	if err := output.Err(); err != nil {
		_ = serverCmd.Process.Kill()
		<-childDone
		return fmt.Errorf("write chat endpoints: %w", err)
	}
	go func() {
		httpDone <- chatServer.ListenAndServe()
	}()

	select {
	case err := <-childDone:
		_ = chatServer.Shutdown(context.Background())
		if err != nil {
			return fmt.Errorf("inference server exited: %w", err)
		}
		return fmt.Errorf("inference server exited")
	case err := <-httpDone:
		_ = serverCmd.Process.Kill()
		<-childDone
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("chat app server failed: %w", err)
	case <-ctx.Done():
		_ = chatServer.Shutdown(context.Background())
		_ = serverCmd.Process.Kill()
		<-childDone
		return ctx.Err()
	}
}

func validateChatOptions(opts chatOptions) (time.Duration, error) {
	if opts.modelPath == "" && !opts.allowUntrained {
		return 0, fmt.Errorf("chat requires --model <checkpoint> or --allow-untrained")
	}
	if opts.port <= 0 || opts.port > 65535 {
		return 0, fmt.Errorf("--port must be between 1 and 65535")
	}
	if opts.apiPort <= 0 || opts.apiPort > 65535 {
		return 0, fmt.Errorf("--api-port must be between 1 and 65535")
	}
	if opts.maxTokens < 0 {
		return 0, fmt.Errorf("--max-tokens must be non-negative")
	}
	if opts.topK < 0 {
		return 0, fmt.Errorf("--top-k must be non-negative")
	}

	readyTimeout, err := time.ParseDuration(opts.readyTimeoutText)
	if err != nil {
		return 0, fmt.Errorf("parse --ready-timeout: %w", err)
	}
	return readyTimeout, nil
}

type chatProxy struct {
	apiBaseURL  string
	modelName   string
	maxTokens   int
	temperature float64
	client      httpDoer
}

func (p chatProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer func() { _ = r.Body.Close() }()

	var payload map[string]any
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64*1024))
	decoder.UseNumber()
	if err := decoder.Decode(&payload); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	if _, ok := payload["messages"]; !ok {
		http.Error(w, "missing messages", http.StatusBadRequest)
		return
	}
	payload["model"] = p.modelName
	if _, ok := payload["max_tokens"]; !ok {
		payload["max_tokens"] = p.maxTokens
	}
	if _, ok := payload["temperature"]; !ok {
		payload["temperature"] = p.temperature
	}

	body, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "encode request", http.StatusInternalServerError)
		return
	}

	client := p.client
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, p.apiBaseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		http.Error(w, "create upstream request", http.StatusInternalServerError)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "inference server unavailable", http.StatusBadGateway)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func serveChatApp(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = io.WriteString(w, chatHTML)
}

func waitForInference(ctx context.Context, url string, timeout time.Duration, childDone <-chan error) (bool, error) {
	client := &http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()

	for {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return false, err
		}
		resp, err := client.Do(req)
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 500 {
				return false, nil
			}
		}

		select {
		case err := <-childDone:
			if err != nil {
				return true, fmt.Errorf("inference server exited before readiness: %w", err)
			}
			return true, fmt.Errorf("inference server exited before readiness")
		case <-deadline.C:
			return false, fmt.Errorf("timed out waiting for inference server at %s", url)
		case <-ticker.C:
		case <-ctx.Done():
			return false, ctx.Err()
		}
	}
}

const chatHTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>zig-nn TinyGPT Chat</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; background: #f6f7f9; color: #17191c; }
    main { max-width: 860px; margin: 0 auto; padding: 24px; display: grid; gap: 16px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    h1 { font-size: 20px; margin: 0; font-weight: 650; }
    #log { min-height: 420px; max-height: 68vh; overflow: auto; background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 16px; display: grid; align-content: start; gap: 12px; }
    .msg { white-space: pre-wrap; padding: 10px 12px; border-radius: 8px; line-height: 1.45; }
    .user { justify-self: end; max-width: 78%; background: #1f6feb; color: white; }
    .assistant { justify-self: start; max-width: 78%; background: #eef1f5; color: #15171a; }
    .error { justify-self: stretch; background: #fff0f0; color: #a40000; border: 1px solid #f0b4b4; }
    form { display: grid; gap: 10px; }
    textarea { min-height: 80px; resize: vertical; padding: 12px; border: 1px solid #c7ced8; border-radius: 8px; font: inherit; }
    .bar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    label { display: inline-flex; gap: 6px; align-items: center; font-size: 13px; }
    input { width: 82px; padding: 6px 8px; border: 1px solid #c7ced8; border-radius: 6px; font: inherit; }
    button { padding: 8px 14px; border: 0; border-radius: 6px; background: #17191c; color: white; font: inherit; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    @media (prefers-color-scheme: dark) {
      body { background: #111316; color: #e8eaed; }
      #log, textarea, input { background: #181b20; border-color: #303640; color: #e8eaed; }
      .assistant { background: #252b33; color: #edf0f4; }
      button { background: #e8eaed; color: #111316; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>zig-nn TinyGPT Chat</h1>
  </header>
  <section id="log" aria-live="polite"></section>
  <form id="chat">
    <textarea id="prompt" placeholder="Type a message..." autofocus></textarea>
    <div class="bar">
      <label>Max tokens <input id="maxTokens" type="number" min="0" max="512" value="120"></label>
      <label>Temperature <input id="temperature" type="number" min="0" max="4" step="0.1" value="1.0"></label>
      <button id="send" type="submit">Send</button>
    </div>
  </form>
</main>
<script>
const log = document.querySelector("#log");
const form = document.querySelector("#chat");
const promptBox = document.querySelector("#prompt");
const send = document.querySelector("#send");
const messages = [];

function addMessage(role, text, error) {
  const node = document.createElement("div");
  node.className = "msg " + (error ? "error" : role);
  node.textContent = text;
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptBox.value.trim();
  if (!text) return;
  promptBox.value = "";
  messages.push({ role: "user", content: text });
  addMessage("user", text);
  send.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages,
        max_tokens: Number(document.querySelector("#maxTokens").value),
        temperature: Number(document.querySelector("#temperature").value)
      })
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error?.message || "request failed");
    }
    const reply = data.choices?.[0]?.message?.content ?? "";
    messages.push({ role: "assistant", content: reply });
    addMessage("assistant", reply || "(empty response)");
  } catch (error) {
    addMessage("assistant", error.message, true);
  } finally {
    send.disabled = false;
    promptBox.focus();
  }
});
</script>
</body>
</html>
`
