<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import {
  Activity,
  Archive,
  Bell,
  ChevronDown,
  CircleGauge,
  Cpu,
  Database,
  Download,
  FileUp,
  Layers,
  PackageCheck,
  Play,
  RefreshCw,
  Settings,
  Terminal,
  Trash2,
  UploadCloud,
  UserRound,
} from "@lucide/vue";

const apiBase = import.meta.env.VITE_API_BASE_URL || "";

const pages = {
  train: { title: "微调模型训练", tab: "训练任务", subtitle: "选择基座模型、上传数据并提交 LoRA 微调任务" },
  rag: { title: "RAG模型配置", tab: "RAG封装", subtitle: "选择基座模型、上传文档并配置问答提示词" },
  base: { title: "基座模型管理", tab: "模型仓库", subtitle: "下载、登记和维护训练可用的基座模型" },
  models: { title: "微调模型管理", tab: "模型产物", subtitle: "管理训练完成后的微调模型" },
};

const currentPage = ref("train");
const baseModels = ref([]);
const downloadJobs = ref([]);
const trainingJobs = ref([]);
const fineTunedModels = ref([]);
const selectedJobId = ref("");
const logContent = ref("");
const toastText = ref("");
let toastTimer = 0;
let pollTimer = 0;

const downloadForm = reactive({
  display_name: "",
  source: "modelscope",
  model_id: "",
  local_path: "",
});

const trainingForm = reactive({
  model_id: "",
  output_name: "",
  domain: "",
  dataset_format: "alpaca",
  training_method: "lora",
  template: "default",
  cutoff_len: 2048,
  num_train_epochs: 3,
  learning_rate: 0.0002,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 8,
  logging_steps: 10,
  save_steps: 500,
  lora_rank: 8,
  lora_alpha: 16,
  lora_dropout: 0.05,
  lora_target: "all",
  fp16: true,
  bf16: false,
});

const ragForm = reactive({
  model_id: "",
  display_name: "",
  domain: "",
  prompt: "",
});

const datasetFile = ref(null);
const ragFile = ref(null);
const datasetFileName = computed(() => datasetFile.value?.name || "选择 JSON / JSONL 文件");
const ragFileName = computed(() => ragFile.value?.name || "选择 TXT / MD / JSON / CSV 文件");
const readyBaseModels = computed(() => baseModels.value.filter((model) => model.status === "ready"));
const selectedJob = computed(() => trainingJobs.value.find((job) => job.id === selectedJobId.value));
const latestDownloadJobs = computed(() => downloadJobs.value.slice(0, 4));
const latestTrainingJobs = computed(() => trainingJobs.value.slice(0, 5));
const ragModels = computed(() => fineTunedModels.value.filter((model) => model.model_type === "rag"));
const latestRagModels = computed(() => ragModels.value.slice(0, 5));

const counters = computed(() => ({
  base: readyBaseModels.value.length,
  jobs: trainingJobs.value.length,
  models: fineTunedModels.value.length,
  rag: ragModels.value.length,
}));

const runningJobs = computed(() => trainingJobs.value.filter((job) => ["queued", "running"].includes(job.status)).length);
const failedJobs = computed(() => trainingJobs.value.filter((job) => ["failed", "interrupted"].includes(job.status)).length);

async function fetchJSON(url, options = {}) {
  const response = await fetch(`${apiBase}${url}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join("；") : payload.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message) {
  toastText.value = message;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toastText.value = "";
  }, 3000);
}

function switchPage(page) {
  currentPage.value = page;
}

async function refreshAll() {
  await Promise.all([loadBaseModels(), loadDownloadJobs(), loadTrainingJobs(), loadFineTunedModels()]);
}

async function loadBaseModels() {
  const payload = await fetchJSON("/api/models/base");
  baseModels.value = payload.items || [];
  if (!trainingForm.model_id && readyBaseModels.value.length) {
    trainingForm.model_id = readyBaseModels.value[0].id;
  }
  if (!ragForm.model_id && readyBaseModels.value.length) {
    ragForm.model_id = readyBaseModels.value[0].id;
  }
}

async function loadDownloadJobs() {
  const payload = await fetchJSON("/api/download-jobs");
  downloadJobs.value = payload.items || [];
}

async function loadTrainingJobs() {
  const payload = await fetchJSON("/api/training/jobs");
  trainingJobs.value = payload.items || [];
  if (selectedJobId.value) {
    await loadLogs(selectedJobId.value);
  }
}

async function loadFineTunedModels() {
  const payload = await fetchJSON("/api/models/finetuned");
  fineTunedModels.value = payload.items || [];
}

async function submitDownload() {
  const payload = {
    display_name: downloadForm.display_name,
    source: downloadForm.source,
    model_id: downloadForm.model_id,
    local_path: downloadForm.source === "local" ? downloadForm.local_path : null,
  };
  await fetchJSON("/api/models/base/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  Object.assign(downloadForm, {
    display_name: "",
    source: "modelscope",
    model_id: "",
    local_path: "",
  });
  showToast("基座模型任务已提交");
  await refreshAll();
}

async function deleteBaseModel(model) {
  if (!window.confirm(`删除基座模型 ${model.display_name}？`)) return;
  await fetchJSON(`/api/models/base/${model.id}`, { method: "DELETE" });
  showToast("基座模型已删除");
  await refreshAll();
}

function onDatasetChange(event) {
  datasetFile.value = event.target.files?.[0] || null;
}

function onRagFileChange(event) {
  ragFile.value = event.target.files?.[0] || null;
}

async function submitTraining() {
  if (!datasetFile.value) {
    showToast("请选择微调数据文件");
    return;
  }
  const form = new FormData();
  Object.entries(trainingForm).forEach(([key, value]) => {
    form.append(key, typeof value === "boolean" ? String(value) : value);
  });
  form.append("dataset_file", datasetFile.value);

  const payload = await fetchJSON("/api/training/jobs", {
    method: "POST",
    body: form,
  });
  selectedJobId.value = payload.job.id;
  datasetFile.value = null;
  document.getElementById("dataset-file").value = "";
  trainingForm.output_name = "";
  trainingForm.domain = "";
  showToast("训练任务已提交");
  await refreshAll();
  await loadLogs(payload.job.id);
}

async function loadLogs(jobId) {
  selectedJobId.value = jobId;
  const response = await fetch(`${apiBase}/api/training/jobs/${jobId}/logs`);
  logContent.value = (await response.text()) || "暂无日志";
}

async function submitRagModel() {
  if (!ragFile.value) {
    showToast("请选择 RAG 数据文件");
    return;
  }
  const form = new FormData();
  Object.entries(ragForm).forEach(([key, value]) => {
    form.append(key, value);
  });
  form.append("knowledge_file", ragFile.value);

  await fetchJSON("/api/rag/models", {
    method: "POST",
    body: form,
  });
  ragFile.value = null;
  document.getElementById("rag-file").value = "";
  Object.assign(ragForm, {
    model_id: readyBaseModels.value[0]?.id || "",
    display_name: "",
    domain: "",
    prompt: "",
  });
  showToast("RAG模型已封装");
  await refreshAll();
}

async function deleteFineTunedModel(model) {
  const modelLabel = model.model_type === "rag" ? "RAG模型" : "微调模型";
  if (!window.confirm(`删除${modelLabel} ${model.display_name}？`)) return;
  await fetchJSON(`/api/models/finetuned/${model.id}`, { method: "DELETE" });
  showToast(`${modelLabel}已删除`);
  await refreshAll();
}

async function deleteTrainingJobHistory(job) {
  if (!window.confirm(`删除训练任务记录 ${job.output_name}？模型文件和微调模型不会被删除。`)) return;
  await fetchJSON(`/api/training/jobs/${job.id}`, { method: "DELETE" });
  if (selectedJobId.value === job.id) {
    selectedJobId.value = "";
    logContent.value = "";
  }
  showToast("训练任务记录已删除");
  await refreshAll();
}

async function deleteDownloadJobHistory(job) {
  if (!window.confirm(`删除下载任务记录 ${job.display_name}？已下载模型不会被删除。`)) return;
  await fetchJSON(`/api/download-jobs/${job.id}`, { method: "DELETE" });
  showToast("下载任务记录已删除");
  await refreshAll();
}

function statusText(status) {
  const map = {
    pending: "等待",
    queued: "排队",
    downloading: "下载中",
    running: "运行中",
    ready: "可用",
    succeeded: "成功",
    failed: "失败",
    interrupted: "中断",
  };
  return map[status] || status || "-";
}

function statusClass(status) {
  if (["ready", "succeeded"].includes(status)) return "success";
  if (["failed", "interrupted"].includes(status)) return "danger";
  if (["pending", "queued", "running", "downloading"].includes(status)) return "processing";
  return "default";
}

function methodText(model) {
  if (model.model_type === "rag" || model.training_method === "rag") return "RAG";
  if (model.training_method === "lora") return "LoRA";
  return model.training_method || "-";
}

function formatDate(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

onMounted(() => {
  refreshAll().catch((error) => showToast(error.message));
  pollTimer = window.setInterval(() => {
    refreshAll().catch(() => {});
  }, 5000);
});

onUnmounted(() => {
  window.clearInterval(pollTimer);
  window.clearTimeout(toastTimer);
});
</script>

<template>
  <div class="app-frame">
    <header class="global-header">
      <div class="platform-brand">
        <div class="platform-mark"></div>
        <strong>智能体构建与管理工具组件</strong>
      </div>
      <div class="header-actions">
        <button class="round-action" type="button" aria-label="通知"><Bell /></button>
        <button class="avatar-action" type="button" aria-label="用户"><UserRound /></button>
      </div>
    </header>

    <div class="app-body">
      <aside class="side-nav">
        <nav class="domain-menu">
          <button class="menu-parent active" type="button">
            <Layers />
            <span>垂直领域大语言模型微调</span>
            <ChevronDown />
          </button>
          <div class="sub-menu">
            <button :class="{ active: currentPage === 'train' }" type="button" @click="switchPage('train')">
              <Activity />
              <span>微调模型训练</span>
            </button>
            <button :class="{ active: currentPage === 'rag' }" type="button" @click="switchPage('rag')">
              <Terminal />
              <span>RAG模型配置</span>
            </button>
            <button :class="{ active: currentPage === 'base' }" type="button" @click="switchPage('base')">
              <Database />
              <span>基座模型管理</span>
            </button>
            <button :class="{ active: currentPage === 'models' }" type="button" @click="switchPage('models')">
              <Archive />
              <span>微调模型管理</span>
            </button>
          </div>
        </nav>

      </aside>

      <main class="content">
        <header class="page-title-row">
          <div>
            <h1>{{ pages[currentPage].title }}</h1>
            <p>{{ pages[currentPage].subtitle }}</p>
          </div>
          <button class="icon-button" type="button" aria-label="刷新" @click="refreshAll">
            <RefreshCw />
          </button>
        </header>

        <section v-if="currentPage === 'train'" class="train-page">
          <section class="guide-card">
            <div class="section-title">操作引导</div>
            <div class="guide-steps">
              <div>
                <div class="guide-icon"><Cpu /></div>
                <strong>选择基座模型</strong>
                <span>从模型仓库选择可用模型</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><UploadCloud /></div>
                <strong>上传微调数据</strong>
                <span>支持 Alpaca / ShareGPT / Messages</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><Settings /></div>
                <strong>配置 LoRA 参数</strong>
                <span>设置训练轮次、学习率和适配器</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><PackageCheck /></div>
                <strong>模型入库</strong>
                <span>训练成功后进入微调模型管理</span>
              </div>
            </div>
          </section>

          <section class="dashboard-grid">
            <div class="metric-card blue-tint">
              <div class="metric-icon"><Database /></div>
              <span>可用基座模型</span>
              <strong>{{ counters.base }}</strong>
            </div>
            <div class="metric-card green-tint">
              <div class="metric-icon"><Activity /></div>
              <span>运行中任务</span>
              <strong>{{ runningJobs }}</strong>
            </div>
            <div class="metric-card orange-tint">
              <div class="metric-icon"><CircleGauge /></div>
              <span>异常任务</span>
              <strong>{{ failedJobs }}</strong>
            </div>
          </section>

          <div class="train-workbench">
            <section class="panel-card train-config">
              <div class="panel-title">
                <h2>训练配置</h2>
                <span class="tag processing">LoRA</span>
              </div>
              <form class="train-form" @submit.prevent="submitTraining">
                <div class="form-grid two">
                  <label>
                    <span>基座模型</span>
                    <select v-model="trainingForm.model_id" required>
                      <option value="">请选择</option>
                      <option v-for="model in readyBaseModels" :key="model.id" :value="model.id">
                        {{ model.display_name }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>输出名称</span>
                    <input v-model="trainingForm.output_name" required placeholder="qwen-customer-lora" />
                  </label>
                  <label>
                    <span>领域</span>
                    <input v-model="trainingForm.domain" placeholder="例如：工业设备运维" />
                  </label>
                  <label>
                    <span>数据格式</span>
                    <select v-model="trainingForm.dataset_format">
                      <option value="alpaca">Alpaca</option>
                      <option value="sharegpt">ShareGPT</option>
                      <option value="openai">OpenAI Messages</option>
                    </select>
                  </label>
                  <label>
                    <span>模板</span>
                    <input v-model="trainingForm.template" />
                  </label>
                </div>

                <label class="upload-field">
                  <input id="dataset-file" type="file" accept=".json,.jsonl" @change="onDatasetChange" />
                  <FileUp />
                  <span>{{ datasetFileName }}</span>
                </label>

                <div class="param-grid">
                  <label>
                    <span>Epoch</span>
                    <input v-model.number="trainingForm.num_train_epochs" type="number" min="0.1" max="100" step="0.1" />
                  </label>
                  <label>
                    <span>学习率</span>
                    <input v-model.number="trainingForm.learning_rate" type="number" min="0.000001" max="1" step="0.000001" />
                  </label>
                  <label>
                    <span>Batch</span>
                    <input v-model.number="trainingForm.per_device_train_batch_size" type="number" min="1" max="512" />
                  </label>
                  <label>
                    <span>累积步</span>
                    <input v-model.number="trainingForm.gradient_accumulation_steps" type="number" min="1" max="1024" />
                  </label>
                  <label>
                    <span>截断长度</span>
                    <input v-model.number="trainingForm.cutoff_len" type="number" min="128" max="32768" step="128" />
                  </label>
                  <label>
                    <span>保存步</span>
                    <input v-model.number="trainingForm.save_steps" type="number" min="1" max="100000" />
                  </label>
                  <label>
                    <span>LoRA Rank</span>
                    <input v-model.number="trainingForm.lora_rank" type="number" min="1" max="1024" />
                  </label>
                  <label>
                    <span>LoRA Alpha</span>
                    <input v-model.number="trainingForm.lora_alpha" type="number" min="1" max="4096" />
                  </label>
                  <label>
                    <span>LoRA Dropout</span>
                    <input v-model.number="trainingForm.lora_dropout" type="number" min="0" max="1" step="0.01" />
                  </label>
                  <label>
                    <span>LoRA Target</span>
                    <input v-model="trainingForm.lora_target" />
                  </label>
                </div>

                <div class="switch-row">
                  <label class="check-line">
                    <input v-model="trainingForm.fp16" type="checkbox" />
                    <span>FP16</span>
                  </label>
                  <label class="check-line">
                    <input v-model="trainingForm.bf16" type="checkbox" />
                    <span>BF16</span>
                  </label>
                </div>

                <div class="form-actions">
                  <button class="primary-button" type="submit">
                    <Play />
                    <span>开始训练</span>
                  </button>
                </div>
              </form>
            </section>

            <section class="panel-card task-list-card">
              <div class="panel-title">
                <h2>训练任务</h2>
                <button class="secondary-button" type="button" @click="loadTrainingJobs"><RefreshCw />刷新</button>
              </div>
              <div class="task-list">
                <article v-for="job in latestTrainingJobs" :key="job.id" class="task-item">
                  <button class="task-main" type="button" @click="loadLogs(job.id)">
                    <span>
                      <strong>{{ job.output_name }}</strong>
                      <small>{{ job.base_model_name }} · {{ job.dataset?.sample_count || 0 }} samples</small>
                      <small v-if="job.domain">领域：{{ job.domain }}</small>
                    </span>
                    <em class="tag" :class="statusClass(job.status)">{{ statusText(job.status) }}</em>
                  </button>
                  <button class="history-delete" type="button" aria-label="删除训练任务记录" @click="deleteTrainingJobHistory(job)">
                    <Trash2 />
                  </button>
                </article>
                <div v-if="!trainingJobs.length" class="empty-state">暂无训练任务</div>
              </div>
            </section>

            <section class="panel-card log-card">
              <div class="panel-title">
                <h2>任务日志</h2>
                <span class="tag">{{ selectedJob?.output_name || "未选择" }}</span>
              </div>
              <pre>{{ logContent || "暂无日志" }}</pre>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'rag'" class="rag-page">
          <section class="dashboard-grid">
            <div class="metric-card blue-tint">
              <div class="metric-icon"><Database /></div>
              <span>可用基座模型</span>
              <strong>{{ counters.base }}</strong>
            </div>
            <div class="metric-card green-tint">
              <div class="metric-icon"><Terminal /></div>
              <span>RAG模型</span>
              <strong>{{ counters.rag }}</strong>
            </div>
            <div class="metric-card orange-tint">
              <div class="metric-icon"><Archive /></div>
              <span>模型产物</span>
              <strong>{{ counters.models }}</strong>
            </div>
          </section>

          <div class="rag-workbench">
            <section class="panel-card rag-config">
              <div class="panel-title">
                <h2>RAG封装配置</h2>
                <span class="tag processing">RAG</span>
              </div>
              <form class="train-form" @submit.prevent="submitRagModel">
                <div class="form-grid two">
                  <label>
                    <span>基座模型</span>
                    <select v-model="ragForm.model_id" required>
                      <option value="">请选择</option>
                      <option v-for="model in readyBaseModels" :key="model.id" :value="model.id">
                        {{ model.display_name }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>模型名称</span>
                    <input v-model="ragForm.display_name" required placeholder="device-manual-rag" />
                  </label>
                  <label>
                    <span>适用领域</span>
                    <input v-model="ragForm.domain" placeholder="例如：设备手册问答" />
                  </label>
                </div>

                <label>
                  <span>提示词</span>
                  <textarea
                    v-model="ragForm.prompt"
                    required
                    maxlength="5000"
                    placeholder="例如：你是工业设备运维助手，请严格依据资料回答，无法确认时说明缺少依据。"
                  ></textarea>
                </label>

                <label class="upload-field">
                  <input id="rag-file" type="file" accept=".txt,.md,.json,.jsonl,.csv,.tsv" @change="onRagFileChange" />
                  <FileUp />
                  <span>{{ ragFileName }}</span>
                </label>

                <div class="form-actions">
                  <button class="primary-button" type="submit">
                    <PackageCheck />
                    <span>生成RAG模型</span>
                  </button>
                </div>
              </form>
            </section>

            <section class="panel-card">
              <div class="panel-title">
                <h2>最近RAG模型</h2>
                <button class="secondary-button" type="button" @click="loadFineTunedModels"><RefreshCw />刷新</button>
              </div>
              <div class="task-list">
                <article v-for="model in latestRagModels" :key="model.id" class="task-item">
                  <div class="task-main">
                    <span>
                      <strong>{{ model.display_name }}</strong>
                      <small>{{ model.base_model_name }} · {{ model.rag?.chunk_count || 0 }} chunks</small>
                      <small v-if="model.rag?.document_name">文档：{{ model.rag.document_name }}</small>
                    </span>
                    <em class="tag" :class="statusClass(model.status)">{{ statusText(model.status) }}</em>
                  </div>
                  <button class="history-delete" type="button" aria-label="删除RAG模型" @click="deleteFineTunedModel(model)">
                    <Trash2 />
                  </button>
                </article>
                <div v-if="!ragModels.length" class="empty-state">暂无RAG模型</div>
              </div>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'base'" class="page-stack">
          <section class="download-hero">
            <div>
              <div class="section-title">模型下载与登记</div>
              <h2>构建可复用的基座模型仓库</h2>
            </div>
            <form class="download-form" @submit.prevent="submitDownload">
              <label>
                <span>模型名称</span>
                <input v-model="downloadForm.display_name" required placeholder="Qwen3-4B" />
              </label>
              <label>
                <span>模型来源</span>
                <select v-model="downloadForm.source">
                  <option value="modelscope">ModelScope</option>
                  <option value="huggingface">Hugging Face</option>
                  <option value="local">本地路径</option>
                </select>
              </label>
              <label>
                <span>模型 ID</span>
                <input v-model="downloadForm.model_id" required placeholder="Qwen/Qwen3-4B" />
              </label>
              <label v-if="downloadForm.source === 'local'">
                <span>本地路径</span>
                <input v-model="downloadForm.local_path" required placeholder="D:\\Models\\Qwen3-4B" />
              </label>
              <button class="primary-button" type="submit">
                <Download />
                <span>提交下载</span>
              </button>
            </form>
          </section>

          <div class="base-grid">
            <section class="panel-card">
              <div class="panel-title">
                <h2>基座模型列表</h2>
                <button class="secondary-button" type="button" @click="loadBaseModels"><RefreshCw />刷新</button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>模型名称</th>
                      <th>来源</th>
                      <th>状态</th>
                      <th>路径</th>
                      <th>创建时间</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!baseModels.length">
                      <td colspan="6" class="empty">暂无数据</td>
                    </tr>
                    <tr v-for="model in baseModels" :key="model.id">
                      <td>
                        <strong>{{ model.display_name }}</strong>
                        <small>{{ model.model_id }}</small>
                      </td>
                      <td>{{ model.source }}</td>
                      <td><span class="tag" :class="statusClass(model.status)">{{ statusText(model.status) }}</span></td>
                      <td class="mono">{{ model.path }}</td>
                      <td>{{ formatDate(model.created_at) }}</td>
                      <td class="action-col">
                        <button class="text-button danger-text" type="button" @click="deleteBaseModel(model)">
                          <Trash2 />
                          <span>删除</span>
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section class="panel-card">
              <div class="panel-title">
                <h2>下载任务</h2>
              </div>
              <div class="download-job-list">
                <div v-for="job in latestDownloadJobs" :key="job.id" class="download-job">
                  <div>
                    <strong>{{ job.display_name }}</strong>
                    <small>{{ formatDate(job.created_at) }}</small>
                    <p v-if="job.error">{{ job.error }}</p>
                  </div>
                  <div class="job-actions">
                    <span class="tag" :class="statusClass(job.status)">{{ statusText(job.status) }}</span>
                    <button class="history-delete" type="button" aria-label="删除下载任务记录" @click="deleteDownloadJobHistory(job)">
                      <Trash2 />
                    </button>
                  </div>
                </div>
                <div v-if="!downloadJobs.length" class="empty-state">暂无下载任务</div>
              </div>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'models'" class="page-stack">
          <section class="panel-card">
            <div class="panel-title">
              <h2>微调模型列表</h2>
              <button class="secondary-button" type="button" @click="loadFineTunedModels"><RefreshCw />刷新</button>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>模型名称</th>
                    <th>适用领域</th>
                    <th>基座模型</th>
                    <th>方法</th>
                    <th>状态</th>
                    <th>路径</th>
                    <th class="action-col">操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-if="!fineTunedModels.length">
                    <td colspan="7" class="empty">暂无数据</td>
                  </tr>
                  <tr v-for="model in fineTunedModels" :key="model.id">
                    <td><strong>{{ model.display_name }}</strong></td>
                    <td>{{ model.domain || "-" }}</td>
                    <td>{{ model.base_model_name }}</td>
                    <td>
                      <span class="tag" :class="model.model_type === 'rag' ? 'processing' : 'default'">{{ methodText(model) }}</span>
                      <small v-if="model.model_type === 'rag'">
                        {{ model.rag?.document_name || "知识文档" }} · {{ model.rag?.chunk_count || 0 }} chunks
                      </small>
                    </td>
                    <td><span class="tag" :class="statusClass(model.status)">{{ statusText(model.status) }}</span></td>
                    <td class="mono">{{ model.path }}</td>
                    <td class="action-col">
                      <button class="text-button danger-text" type="button" @click="deleteFineTunedModel(model)">
                        <Trash2 />
                        <span>删除</span>
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </main>
    </div>

    <div v-if="toastText" class="message">{{ toastText }}</div>
  </div>
</template>
