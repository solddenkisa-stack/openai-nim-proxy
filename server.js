// server.js - OpenAI to NVIDIA NIM API Proxy (with auto model rotation)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// 🔄 Model rotation pool - tried in order when a 429 is hit
const MODEL_POOL = [
  'openai/gpt-oss-120b',
  'deepseek-ai/deepseek-v3.2',
  'z-ai/glm5',
  'moonshotai/kimi-k2.6',
  'meta/llama-4-maverick-17b-128e-instruct',
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'meta/llama-3.1-405b-instruct',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'deepseek-ai/deepseek-v3.1',
  'qwen/qwen3-235b-a22b',
];

// Track which index we're currently on
let currentModelIndex = 0;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'z-ai/glm5',
};

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    current_model: MODEL_POOL[currentModelIndex],
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1', (req, res) => {
  res.json({ status: 'ok', message: 'OpenAI NIM Proxy v1 ready' });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Helper: attempt a single NIM request
async function attemptRequest(nimModel, messages, temperature, max_tokens, stream) {
  const nimRequest = {
    model: nimModel,
    messages,
    temperature: temperature || 0.6,
    max_tokens: max_tokens || 9024,
    extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
    stream: stream || false
  };

  return axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
    headers: {
      'Authorization': `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    responseType: stream ? 'stream' : 'json',
    validateStatus: s => true // handle status codes manually
  });
}

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Resolve starting model
    let startingModel = MODEL_MAPPING[model] || MODEL_POOL[currentModelIndex];

    // Build attempt order: start with mapped/current, then rotate through pool
    const poolWithoutStart = MODEL_POOL.filter(m => m !== startingModel);
    const attemptOrder = [startingModel, ...poolWithoutStart];

    let response = null;
    let usedModel = null;

    for (let i = 0; i < attemptOrder.length; i++) {
      const nimModel = attemptOrder[i];
      console.log(`Trying model: ${nimModel}`);

      response = await attemptRequest(nimModel, messages, temperature, max_tokens, stream);

      if (response.status === 429) {
        console.warn(`429 on ${nimModel}, rotating...`);
        // Advance global index so next request starts on a fresh model
        currentModelIndex = (currentModelIndex + 1) % MODEL_POOL.length;
        continue;
      }

      if (response.status >= 200 && response.status < 300) {
        usedModel = nimModel;
        break;
      }

      // Other errors - stop trying
      break;
    }

    if (!usedModel || !response) {
      return res.status(429).json({
        error: {
          message: 'All models are rate limited. Please wait a moment and try again.',
          type: 'rate_limit_error',
          code: 429
        }
      });
    }

    console.log(`Using model: ${usedModel}`);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) { res.write(line + '\n'); return; }
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combined = '';
                  if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
                  else if (reasoning) { combined = reasoning; }
                  if (content && reasoningStarted) { combined += '</think>\n\n' + content; reasoningStarted = false; }
                  else if (content) { combined += content; }
                  if (combined) { data.choices[0].delta.content = combined; delete data.choices[0].delta.reasoning_content; }
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) { res.write(line + '\n'); }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => { console.error('Stream error:', err); res.end(); });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Model pool: ${MODEL_POOL.length} models available`);
  console.log(`Starting model: ${MODEL_POOL[currentModelIndex]}`);
});
