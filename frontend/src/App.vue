<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import {
  Activity,
  Archive,
  Bell,
  ChevronDown,
  CircleGauge,
  Copy,
  Cpu,
  Database,
  Download,
  FileUp,
  Layers,
  PackageCheck,
  Play,
  Plus,
  Power,
  RefreshCw,
  Search,
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
  workflow: { title: "智能体工作流构建", tab: "工作流", subtitle: "按顺序添加节点，构建从输入到输出的智能体链路" },
  resources: { title: "资源库管理", tab: "资源资产", subtitle: "管理智能体可复用的知识源与提示词" },
  agentManage: { title: "智能体管理", tab: "智能体", subtitle: "管理已保存的智能体并测试运行效果" },
};

const currentPage = ref("train");
const baseModels = ref([]);
const downloadJobs = ref([]);
const trainingJobs = ref([]);
const fineTunedModels = ref([]);
const knowledgeSources = ref([]);
const promptAssets = ref([]);
const agentWorkflows = ref([]);
const agentServices = ref([]);
const selectedJobId = ref("");
const logContent = ref("");
const toastText = ref("");
const trainingError = ref("");
const trainingSubmitting = ref(false);
const resourceTab = ref("knowledge");
const resourceSearch = ref("");
const showKnowledgeForm = ref(false);
const showPromptForm = ref(false);
const selectedKnowledgeIds = ref([]);
const selectedPromptIds = ref([]);
const selectedWorkflowNodeId = ref("input-node");
const workflowTestInput = ref("");
const workflowTestResult = ref(null);
const workflowRunning = ref(false);
const managedAgentId = ref("");
const managedAgentTestInput = ref("");
const managedAgentTestResult = ref(null);
const managedAgentRunning = ref(false);
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

const knowledgeForm = reactive({
  name: "",
  source_type: "text",
  description: "",
});

const promptForm = reactive({
  name: "",
  prompt_type: "system",
  description: "",
  content: "",
});

const workflowForm = reactive({
  name: "",
  description: "",
  nodes: createDefaultWorkflowNodes(),
});

const datasetFile = ref(null);
const ragFile = ref(null);
const knowledgeFile = ref(null);
const promptFile = ref(null);
const datasetFileName = computed(() => datasetFile.value?.name || "选择 JSON / JSONL 文件");
const ragFileName = computed(() => ragFile.value?.name || "选择 TXT / MD / JSON / CSV 文件");
const knowledgeFileName = computed(() => knowledgeFile.value?.name || "选择知识源文件");
const promptFileName = computed(() => promptFile.value?.name || "可选：上传 .txt 提示词文件");
const readyBaseModels = computed(() => baseModels.value.filter((model) => model.status === "ready"));
const selectedJob = computed(() => trainingJobs.value.find((job) => job.id === selectedJobId.value));
const latestDownloadJobs = computed(() => downloadJobs.value.slice(0, 4));
const latestTrainingJobs = computed(() => trainingJobs.value.slice(0, 5));
const ragModels = computed(() => fineTunedModels.value.filter((model) => model.model_type === "rag"));
const trainableFineTunedModels = computed(() => fineTunedModels.value);
const latestRagModels = computed(() => ragModels.value.slice(0, 5));
const filteredKnowledgeSources = computed(() => filterResources(knowledgeSources.value));
const filteredPromptAssets = computed(() => filterResources(promptAssets.value));
const selectedWorkflowNode = computed(() => workflowForm.nodes.find((node) => node.id === selectedWorkflowNodeId.value));
const managedAgent = computed(() => agentWorkflows.value.find((agent) => agent.id === managedAgentId.value));
const runningAgentServices = computed(() => agentServices.value.filter((service) => service.status === "running"));
const agentServiceMap = computed(() => {
  const map = new Map();
  runningAgentServices.value.forEach((service) => {
    if (!map.has(service.agent_id)) {
      map.set(service.agent_id, service);
    }
  });
  return map;
});
const allFilteredKnowledgeSelected = computed(
  () => filteredKnowledgeSources.value.length > 0 && filteredKnowledgeSources.value.every((item) => selectedKnowledgeIds.value.includes(item.id)),
);
const allFilteredPromptSelected = computed(
  () => filteredPromptAssets.value.length > 0 && filteredPromptAssets.value.every((item) => selectedPromptIds.value.includes(item.id)),
);

const counters = computed(() => ({
  base: readyBaseModels.value.length,
  jobs: trainingJobs.value.length,
  models: fineTunedModels.value.length,
  rag: ragModels.value.length,
  knowledge: knowledgeSources.value.length,
  prompts: promptAssets.value.length,
  agents: agentWorkflows.value.length,
}));

const runningJobs = computed(() => trainingJobs.value.filter((job) => ["queued", "running"].includes(job.status)).length);
const failedJobs = computed(() => trainingJobs.value.filter((job) => ["failed", "interrupted"].includes(job.status)).length);

async function fetchJSON(url, options = {}) {
  const response = await fetch(`${apiBase}${url}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = formatErrorDetail(payload.detail) || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        const field = Array.isArray(item.loc) ? item.loc.join(".") : "";
        return field ? `${field}：${item.msg}` : item.msg;
      })
      .filter(Boolean)
      .join("；");
  }
  return detail ? String(detail) : "";
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
  await Promise.all([
    loadBaseModels(),
    loadDownloadJobs(),
    loadTrainingJobs(),
    loadFineTunedModels(),
    loadKnowledgeSources(),
    loadPromptAssets(),
    loadAgentWorkflows(),
    loadAgentServices(),
  ]);
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

async function loadKnowledgeSources() {
  const payload = await fetchJSON("/api/resources/knowledge-sources");
  knowledgeSources.value = payload.items || [];
  selectedKnowledgeIds.value = selectedKnowledgeIds.value.filter((id) => knowledgeSources.value.some((item) => item.id === id));
}

async function loadPromptAssets() {
  const payload = await fetchJSON("/api/resources/prompts");
  promptAssets.value = payload.items || [];
  selectedPromptIds.value = selectedPromptIds.value.filter((id) => promptAssets.value.some((item) => item.id === id));
}

async function loadAgentWorkflows() {
  const payload = await fetchJSON("/api/agents");
  agentWorkflows.value = payload.items || [];
  if (managedAgentId.value && !agentWorkflows.value.some((agent) => agent.id === managedAgentId.value)) {
    managedAgentId.value = "";
    managedAgentTestResult.value = null;
  }
}

async function loadAgentServices() {
  const payload = await fetchJSON("/api/agent-services");
  agentServices.value = payload.items || [];
}

async function refreshAgentManagement() {
  await Promise.all([loadAgentWorkflows(), loadAgentServices()]);
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
  trainingError.value = "";
}

function onRagFileChange(event) {
  ragFile.value = event.target.files?.[0] || null;
}

function onKnowledgeFileChange(event) {
  knowledgeFile.value = event.target.files?.[0] || null;
}

async function onPromptFileChange(event) {
  const file = event.target.files?.[0] || null;
  if (!file) {
    promptFile.value = null;
    return;
  }

  if (!file.name.toLowerCase().endsWith(".txt")) {
    promptFile.value = null;
    event.target.value = "";
    showToast("提示词模板文件仅支持 .txt");
    return;
  }

  try {
    promptForm.content = await file.text();
    promptFile.value = file;
    if (!promptForm.name.trim()) {
      promptForm.name = file.name.replace(/\.[^.]+$/, "");
    }
    showToast("已读取提示词文件内容");
  } catch {
    promptFile.value = null;
    event.target.value = "";
    showToast("提示词文件读取失败");
  }
}

function filterResources(items) {
  const keyword = resourceSearch.value.trim().toLowerCase();
  if (!keyword) return items;
  return items.filter((item) =>
    [item.name, item.description, item.type_label, item.original_name, item.preview]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(keyword)),
  );
}

function createDefaultWorkflowNodes() {
  return [
    { id: "input-node", type: "input", label: "用户输入", config: { variable: "input" } },
    { id: "output-node", type: "output", label: "结果输出", config: { output_key: "output" } },
  ];
}

function createWorkflowNode(type) {
  const id = `${type}-${Date.now()}-${Math.random().toString(16).slice(2, 7)}`;
  const labels = {
    input: "用户输入",
    output: "结果输出",
    llm: "大模型",
    finetuned: "微调模型",
    knowledge: "知识库",
  };
  const config = {
    input: { variable: "input" },
    output: { output_key: "output" },
    llm: {
      model_id: readyBaseModels.value[0]?.id || "",
      prompt_asset_id: "",
      prompt: "",
      filter_thinking: true,
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.9,
    },
    finetuned: {
      fine_tuned_model_id: trainableFineTunedModels.value[0]?.id || "",
      model_id: readyBaseModels.value[0]?.id || "",
      prompt_asset_id: "",
      prompt: "",
      filter_thinking: true,
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.9,
      top_k: 4,
    },
    knowledge: { knowledge_source_id: knowledgeSources.value[0]?.id || "" },
  };
  return { id, type, label: labels[type], config: config[type] };
}

function addWorkflowNode(type) {
  const node = createWorkflowNode(type);
  const outputIndex = workflowForm.nodes.findIndex((item) => item.type === "output");
  if (outputIndex >= 0 && type !== "output") {
    workflowForm.nodes.splice(outputIndex, 0, node);
  } else {
    workflowForm.nodes.push(node);
  }
  selectedWorkflowNodeId.value = node.id;
}

function removeWorkflowNode(nodeId) {
  const index = workflowForm.nodes.findIndex((node) => node.id === nodeId);
  if (index < 0) return;
  if (workflowForm.nodes[index].type === "input" && index === 0) {
    showToast("首个用户输入节点不可删除");
    return;
  }
  if (workflowForm.nodes[index].type === "output" && index === workflowForm.nodes.length - 1) {
    showToast("末尾结果输出节点不可删除");
    return;
  }
  workflowForm.nodes.splice(index, 1);
  selectedWorkflowNodeId.value = workflowForm.nodes[Math.max(0, index - 1)]?.id || "input-node";
}

function resetWorkflowBuilder() {
  Object.assign(workflowForm, {
    name: "",
    description: "",
    nodes: createDefaultWorkflowNodes(),
  });
  selectedWorkflowNodeId.value = "input-node";
  workflowTestInput.value = "";
  workflowTestResult.value = null;
}

function loadAgentToBuilder(agent) {
  Object.assign(workflowForm, {
    name: agent.name,
    description: agent.description || "",
    nodes: JSON.parse(JSON.stringify(agent.nodes || createDefaultWorkflowNodes())),
  });
  selectedWorkflowNodeId.value = workflowForm.nodes[0]?.id || "input-node";
  workflowTestResult.value = null;
  currentPage.value = "workflow";
}

function workflowNodeSummary(node) {
  if (node.type === "input") return "接收用户输入";
  if (node.type === "output") return "返回最终结果";
  if (node.type === "knowledge") {
    const source = knowledgeSources.value.find((item) => item.id === node.config.knowledge_source_id);
    return source ? `加载：${source.name}` : "选择知识源";
  }
  if (node.type === "llm") {
    const model = baseModels.value.find((item) => item.id === node.config.model_id);
    return model ? `基座：${model.display_name}` : "选择基座模型";
  }
  if (node.type === "finetuned") {
    const model = fineTunedModels.value.find((item) => item.id === node.config.fine_tuned_model_id);
    const base = baseModels.value.find((item) => item.id === node.config.model_id);
    if (model) {
      return `${methodText(model)}：${model.display_name}`;
    }
    return base ? `基座：${base.display_name}` : "选择模型";
  }
  return "-";
}

function applyPromptTemplateToNode(node, promptAssetId) {
  if (!node?.config) return;
  node.config.prompt_asset_id = promptAssetId;
  const asset = promptAssets.value.find((item) => item.id === promptAssetId);
  if (asset) {
    node.config.prompt = asset.content || "";
  }
}

function agentServiceFor(agent) {
  return agentServiceMap.value.get(agent.id);
}

async function submitTraining() {
  trainingError.value = "";
  if (!datasetFile.value) {
    showToast("请选择微调数据文件");
    trainingError.value = "请选择微调数据文件。";
    return;
  }
  const form = new FormData();
  Object.entries(trainingForm).forEach(([key, value]) => {
    form.append(key, typeof value === "boolean" ? String(value) : value);
  });
  form.append("dataset_file", datasetFile.value);

  trainingSubmitting.value = true;
  try {
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
  } catch (error) {
    trainingError.value = error.message || "训练任务提交失败。";
    showToast(trainingError.value);
  } finally {
    trainingSubmitting.value = false;
  }
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

async function submitKnowledgeSource() {
  if (!knowledgeFile.value) {
    showToast("请选择知识源文件");
    return;
  }
  const form = new FormData();
  Object.entries(knowledgeForm).forEach(([key, value]) => {
    form.append(key, value);
  });
  form.append("knowledge_file", knowledgeFile.value);

  await fetchJSON("/api/resources/knowledge-sources", {
    method: "POST",
    body: form,
  });
  knowledgeFile.value = null;
  document.getElementById("knowledge-file").value = "";
  Object.assign(knowledgeForm, {
    name: "",
    source_type: "text",
    description: "",
  });
  showKnowledgeForm.value = false;
  showToast("知识源已添加");
  await loadKnowledgeSources();
}

async function submitPromptAsset() {
  if (!promptForm.content.trim()) {
    showToast("请填写提示词内容或上传 .txt 提示词文件");
    return;
  }
  const form = new FormData();
  Object.entries(promptForm).forEach(([key, value]) => {
    form.append(key, value);
  });

  await fetchJSON("/api/resources/prompts", {
    method: "POST",
    body: form,
  });
  promptFile.value = null;
  document.getElementById("prompt-file").value = "";
  Object.assign(promptForm, {
    name: "",
    prompt_type: "system",
    description: "",
    content: "",
  });
  showPromptForm.value = false;
  showToast("提示词已添加");
  await loadPromptAssets();
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

async function deleteKnowledgeSource(item) {
  if (!window.confirm(`删除知识源 ${item.name}？`)) return;
  await fetchJSON(`/api/resources/knowledge-sources/${item.id}`, { method: "DELETE" });
  selectedKnowledgeIds.value = selectedKnowledgeIds.value.filter((id) => id !== item.id);
  showToast("知识源已删除");
  await loadKnowledgeSources();
}

async function deletePromptAsset(item) {
  if (!window.confirm(`删除提示词 ${item.name}？`)) return;
  await fetchJSON(`/api/resources/prompts/${item.id}`, { method: "DELETE" });
  selectedPromptIds.value = selectedPromptIds.value.filter((id) => id !== item.id);
  showToast("提示词已删除");
  await loadPromptAssets();
}

function toggleKnowledgeSelection(id) {
  selectedKnowledgeIds.value = selectedKnowledgeIds.value.includes(id)
    ? selectedKnowledgeIds.value.filter((itemId) => itemId !== id)
    : [...selectedKnowledgeIds.value, id];
}

function togglePromptSelection(id) {
  selectedPromptIds.value = selectedPromptIds.value.includes(id)
    ? selectedPromptIds.value.filter((itemId) => itemId !== id)
    : [...selectedPromptIds.value, id];
}

function setAllKnowledgeSelection(event) {
  selectedKnowledgeIds.value = event.target.checked ? filteredKnowledgeSources.value.map((item) => item.id) : [];
}

function setAllPromptSelection(event) {
  selectedPromptIds.value = event.target.checked ? filteredPromptAssets.value.map((item) => item.id) : [];
}

async function batchDeleteKnowledgeSources() {
  if (!selectedKnowledgeIds.value.length) {
    showToast("请选择知识源");
    return;
  }
  if (!window.confirm(`删除选中的 ${selectedKnowledgeIds.value.length} 个知识源？`)) return;
  await fetchJSON("/api/resources/knowledge-sources/batch-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: selectedKnowledgeIds.value }),
  });
  selectedKnowledgeIds.value = [];
  showToast("知识源已批量删除");
  await loadKnowledgeSources();
}

async function batchDeletePromptAssets() {
  if (!selectedPromptIds.value.length) {
    showToast("请选择提示词");
    return;
  }
  if (!window.confirm(`删除选中的 ${selectedPromptIds.value.length} 个提示词？`)) return;
  await fetchJSON("/api/resources/prompts/batch-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: selectedPromptIds.value }),
  });
  selectedPromptIds.value = [];
  showToast("提示词已批量删除");
  await loadPromptAssets();
}

async function saveAgentWorkflow() {
  if (!workflowForm.name.trim()) {
    showToast("请填写智能体名称");
    return;
  }
  const payload = {
    name: workflowForm.name,
    description: workflowForm.description,
    nodes: workflowForm.nodes,
  };
  const response = await fetchJSON("/api/agents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  showToast("智能体已保存");
  await loadAgentWorkflows();
  managedAgentId.value = response.item.id;
}

async function runWorkflowPreview() {
  const payload = {
    agent: {
      name: workflowForm.name || "预览智能体",
      description: workflowForm.description,
      nodes: workflowForm.nodes,
    },
    input_text: workflowTestInput.value,
  };
  workflowRunning.value = true;
  workflowTestResult.value = { output: "模型启动与推理中..." };
  try {
    workflowTestResult.value = await fetchJSON("/api/agents/preview-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    workflowTestResult.value = { output: `运行失败：${error.message}` };
    showToast(error.message);
  } finally {
    workflowRunning.value = false;
  }
}

async function runManagedAgent(agent = managedAgent.value) {
  if (!agent) {
    showToast("请选择智能体");
    return;
  }
  managedAgentId.value = agent.id;
  managedAgentRunning.value = true;
  managedAgentTestResult.value = { output: "模型启动与推理中..." };
  try {
    managedAgentTestResult.value = await fetchJSON(`/api/agents/${agent.id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_text: managedAgentTestInput.value }),
    });
  } catch (error) {
    managedAgentTestResult.value = { output: `运行失败：${error.message}` };
    showToast(error.message);
  } finally {
    managedAgentRunning.value = false;
  }
}

async function startAgentService(agent) {
  const payload = await fetchJSON(`/api/agents/${agent.id}/service`, { method: "POST" });
  showToast("智能体服务已开启");
  await loadAgentServices();
  return payload.service;
}

async function stopAgentService(service) {
  if (!service) return;
  await fetchJSON(`/api/agent-services/${service.id}`, { method: "DELETE" });
  showToast("智能体服务已停止");
  await loadAgentServices();
}

async function toggleAgentService(agent, event) {
  const checked = event.target.checked;
  const service = agentServiceFor(agent);
  try {
    if (checked) {
      await startAgentService(agent);
    } else {
      await stopAgentService(service);
    }
  } catch (error) {
    event.target.checked = Boolean(service);
    showToast(error.message);
  }
}

async function copyServiceUrl(service) {
  if (!service?.url) return;
  try {
    await navigator.clipboard.writeText(service.url);
    showToast("服务 URL 已复制");
  } catch {
    const input = document.createElement("textarea");
    input.value = service.url;
    document.body.appendChild(input);
    input.select();
    document.execCommand("copy");
    document.body.removeChild(input);
    showToast("服务 URL 已复制");
  }
}

async function deleteAgentWorkflow(agent) {
  if (!window.confirm(`删除智能体 ${agent.name}？`)) return;
  await fetchJSON(`/api/agents/${agent.id}`, { method: "DELETE" });
  if (managedAgentId.value === agent.id) {
    managedAgentId.value = "";
    managedAgentTestResult.value = null;
  }
  showToast("智能体已删除");
  await Promise.all([loadAgentWorkflows(), loadAgentServices()]);
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

function formatSize(value) {
  if (!value) return "-";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function statsText(item) {
  if (item.stats?.rows) return `${item.stats.rows} 行`;
  if (item.stats?.items) return `${item.stats.items} 项`;
  if (item.stats?.lines) return `${item.stats.lines} 行`;
  if (item.stats?.chars) return `${item.stats.chars} 字符`;
  return formatSize(item.size_bytes);
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
          <button class="menu-parent" :class="{ active: ['train', 'rag', 'base', 'models'].includes(currentPage) }" type="button">
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

        <nav class="domain-menu">
          <button class="menu-parent" :class="{ active: ['workflow', 'resources', 'agentManage'].includes(currentPage) }" type="button">
            <PackageCheck />
            <span>智能体构建与管理</span>
            <ChevronDown />
          </button>
          <div class="sub-menu">
            <button :class="{ active: currentPage === 'workflow' }" type="button" @click="switchPage('workflow')">
              <Activity />
              <span>智能体工作流构建</span>
            </button>
            <button :class="{ active: currentPage === 'resources' }" type="button" @click="switchPage('resources')">
              <Database />
              <span>资源库管理</span>
            </button>
            <button :class="{ active: currentPage === 'agentManage' }" type="button" @click="switchPage('agentManage')">
              <Archive />
              <span>智能体管理</span>
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
                <div v-if="trainingError" class="form-error">
                  {{ trainingError }}
                </div>

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
                  <button class="primary-button" type="submit" :disabled="trainingSubmitting">
                    <Play />
                    <span>{{ trainingSubmitting ? "提交中" : "开始训练" }}</span>
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

        <section v-if="currentPage === 'workflow'" class="workflow-page">
          <section class="workflow-header panel-card">
            <div class="form-grid workflow-name-grid">
              <label>
                <span>智能体名称</span>
                <input v-model="workflowForm.name" required placeholder="设备运维助手" />
              </label>
              <label>
                <span>描述</span>
                <input v-model="workflowForm.description" placeholder="顺序执行的智能体工作流" />
              </label>
            </div>
            <div class="workflow-header-actions">
              <button class="secondary-button" type="button" @click="resetWorkflowBuilder">
                <RefreshCw />
                <span>重置</span>
              </button>
              <button class="primary-button" type="button" @click="saveAgentWorkflow">
                <PackageCheck />
                <span>保存智能体</span>
              </button>
            </div>
          </section>

          <section class="workflow-canvas panel-card">
            <div class="workflow-canvas-grid">
              <template v-for="(node, index) in workflowForm.nodes" :key="node.id">
                <button class="workflow-node" :class="{ active: selectedWorkflowNodeId === node.id }" type="button" @click="selectedWorkflowNodeId = node.id">
                  <span class="workflow-node-icon" :class="`node-${node.type}`">
                    <Database v-if="node.type === 'knowledge'" />
                    <Cpu v-else-if="node.type === 'llm' || node.type === 'finetuned'" />
                    <Terminal v-else-if="node.type === 'output'" />
                    <Activity v-else />
                  </span>
                  <strong>{{ node.label }}</strong>
                  <small>{{ workflowNodeSummary(node) }}</small>
                </button>
                <div v-if="index < workflowForm.nodes.length - 1" class="workflow-link">
                  <i></i>
                </div>
              </template>
            </div>
          </section>

          <div class="workflow-builder-grid">
            <section class="panel-card">
              <div class="panel-title">
                <h2>添加节点</h2>
              </div>
              <div class="node-palette">
                <button type="button" @click="addWorkflowNode('input')">
                  <Activity />
                  <span>用户输入</span>
                </button>
                <button type="button" @click="addWorkflowNode('llm')">
                  <Cpu />
                  <span>大模型</span>
                </button>
                <button type="button" @click="addWorkflowNode('finetuned')">
                  <Archive />
                  <span>微调模型</span>
                </button>
                <button type="button" @click="addWorkflowNode('knowledge')">
                  <Database />
                  <span>知识库</span>
                </button>
                <button type="button" @click="addWorkflowNode('output')">
                  <Terminal />
                  <span>结果输出</span>
                </button>
              </div>
            </section>

            <section class="panel-card">
              <div class="panel-title">
                <h2>节点配置</h2>
                <button v-if="selectedWorkflowNode" class="text-button danger-text" type="button" @click="removeWorkflowNode(selectedWorkflowNode.id)">
                  <Trash2 />
                  <span>删除节点</span>
                </button>
              </div>
              <div v-if="selectedWorkflowNode" class="node-config-form">
                <label>
                  <span>节点名称</span>
                  <input v-model="selectedWorkflowNode.label" />
                </label>

                <template v-if="selectedWorkflowNode.type === 'input'">
                  <label>
                    <span>输入变量</span>
                    <input v-model="selectedWorkflowNode.config.variable" placeholder="input" />
                  </label>
                </template>

                <template v-if="selectedWorkflowNode.type === 'knowledge'">
                  <label>
                    <span>知识源</span>
                    <select v-model="selectedWorkflowNode.config.knowledge_source_id">
                      <option value="">请选择知识源</option>
                      <option v-for="item in knowledgeSources" :key="item.id" :value="item.id">
                        {{ item.name }}
                      </option>
                    </select>
                  </label>
                </template>

                <template v-if="selectedWorkflowNode.type === 'llm'">
                  <label>
                    <span>基座模型</span>
                    <select v-model="selectedWorkflowNode.config.model_id">
                      <option value="">请选择基座模型</option>
                      <option v-for="model in readyBaseModels" :key="model.id" :value="model.id">
                        {{ model.display_name }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>提示词模板</span>
                    <select
                      :value="selectedWorkflowNode.config.prompt_asset_id || ''"
                      @change="applyPromptTemplateToNode(selectedWorkflowNode, $event.target.value)"
                    >
                      <option value="">手动输入 / 不使用模板</option>
                      <option v-for="item in promptAssets" :key="item.id" :value="item.id">
                        {{ item.name }} · {{ item.type_label }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>提示词</span>
                    <textarea v-model="selectedWorkflowNode.config.prompt" placeholder="请输入该节点的大模型提示词"></textarea>
                  </label>
                  <div class="form-grid two">
                    <label>
                      <span>最大输出 Token</span>
                      <input v-model.number="selectedWorkflowNode.config.max_new_tokens" type="number" min="1" max="4096" />
                    </label>
                    <label>
                      <span>温度</span>
                      <input v-model.number="selectedWorkflowNode.config.temperature" type="number" min="0" max="2" step="0.1" />
                    </label>
                    <label>
                      <span>Top P</span>
                      <input v-model.number="selectedWorkflowNode.config.top_p" type="number" min="0.01" max="1" step="0.01" />
                    </label>
                  </div>
                  <label class="check-line">
                    <input v-model="selectedWorkflowNode.config.filter_thinking" type="checkbox" />
                    <span>过滤思考内容</span>
                  </label>
                </template>

                <template v-if="selectedWorkflowNode.type === 'finetuned'">
                  <label>
                    <span>微调模型</span>
                    <select v-model="selectedWorkflowNode.config.fine_tuned_model_id">
                      <option value="">未选择时使用基座模型</option>
                      <option v-for="model in trainableFineTunedModels" :key="model.id" :value="model.id">
                        {{ model.display_name }} · {{ methodText(model) }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>基座模型</span>
                    <select v-model="selectedWorkflowNode.config.model_id">
                      <option value="">请选择基座模型</option>
                      <option v-for="model in readyBaseModels" :key="model.id" :value="model.id">
                        {{ model.display_name }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>提示词模板</span>
                    <select
                      :value="selectedWorkflowNode.config.prompt_asset_id || ''"
                      @change="applyPromptTemplateToNode(selectedWorkflowNode, $event.target.value)"
                    >
                      <option value="">手动输入 / 不使用模板</option>
                      <option v-for="item in promptAssets" :key="item.id" :value="item.id">
                        {{ item.name }} · {{ item.type_label }}
                      </option>
                    </select>
                  </label>
                  <label>
                    <span>提示词</span>
                    <textarea v-model="selectedWorkflowNode.config.prompt" placeholder="请输入该节点的微调模型提示词"></textarea>
                  </label>
                  <div class="form-grid two">
                    <label>
                      <span>最大输出 Token</span>
                      <input v-model.number="selectedWorkflowNode.config.max_new_tokens" type="number" min="1" max="4096" />
                    </label>
                    <label>
                      <span>温度</span>
                      <input v-model.number="selectedWorkflowNode.config.temperature" type="number" min="0" max="2" step="0.1" />
                    </label>
                    <label>
                      <span>Top P</span>
                      <input v-model.number="selectedWorkflowNode.config.top_p" type="number" min="0.01" max="1" step="0.01" />
                    </label>
                    <label>
                      <span>RAG 召回数</span>
                      <input v-model.number="selectedWorkflowNode.config.top_k" type="number" min="1" max="12" />
                    </label>
                  </div>
                  <label class="check-line">
                    <input v-model="selectedWorkflowNode.config.filter_thinking" type="checkbox" />
                    <span>过滤思考内容</span>
                  </label>
                </template>

                <template v-if="selectedWorkflowNode.type === 'output'">
                  <label>
                    <span>输出变量</span>
                    <input v-model="selectedWorkflowNode.config.output_key" placeholder="output" />
                  </label>
                </template>
              </div>
            </section>

            <section class="panel-card">
              <div class="panel-title">
                <h2>试运行</h2>
                <button class="secondary-button" type="button" :disabled="workflowRunning" @click="runWorkflowPreview">
                  <Play />
                  <span>{{ workflowRunning ? "运行中" : "运行" }}</span>
                </button>
              </div>
              <div class="agent-test-panel">
                <label>
                  <span>测试输入</span>
                  <textarea v-model="workflowTestInput" placeholder="输入一段用户问题"></textarea>
                </label>
                <pre>{{ workflowTestResult?.output || "暂无运行结果" }}</pre>
              </div>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'resources'" class="resource-page">
          <section class="resource-shell panel-card">
            <div class="resource-toolbar">
              <div class="resource-tabs">
                <button :class="{ active: resourceTab === 'knowledge' }" type="button" @click="resourceTab = 'knowledge'">
                  <Database />
                  <span>知识库</span>
                  <em>{{ counters.knowledge }}</em>
                </button>
                <button :class="{ active: resourceTab === 'prompts' }" type="button" @click="resourceTab = 'prompts'">
                  <Terminal />
                  <span>提示词</span>
                  <em>{{ counters.prompts }}</em>
                </button>
              </div>
              <div class="resource-actions">
                <div class="search-field">
                  <Search />
                  <input v-model="resourceSearch" placeholder="搜索" />
                </div>
                <button
                  v-if="resourceTab === 'knowledge'"
                  class="primary-button"
                  type="button"
                  @click="showKnowledgeForm = !showKnowledgeForm"
                >
                  <Plus />
                  <span>添加知识源</span>
                </button>
                <button v-else class="primary-button" type="button" @click="showPromptForm = !showPromptForm">
                  <Plus />
                  <span>添加提示词</span>
                </button>
              </div>
            </div>

            <form v-if="resourceTab === 'knowledge' && showKnowledgeForm" class="resource-form" @submit.prevent="submitKnowledgeSource">
              <div class="form-grid resource-form-grid">
                <label>
                  <span>知识源名称</span>
                  <input v-model="knowledgeForm.name" required placeholder="设备维护手册" />
                </label>
                <label>
                  <span>知识源类型</span>
                  <select v-model="knowledgeForm.source_type">
                    <option value="text">文本</option>
                    <option value="markdown">Markdown</option>
                    <option value="json">JSON</option>
                    <option value="table">表格</option>
                  </select>
                </label>
                <label>
                  <span>描述</span>
                  <input v-model="knowledgeForm.description" placeholder="用于运维问答" />
                </label>
                <label class="upload-field">
                  <input id="knowledge-file" type="file" accept=".txt,.md,.json,.jsonl,.csv,.tsv" @change="onKnowledgeFileChange" />
                  <FileUp />
                  <span>{{ knowledgeFileName }}</span>
                </label>
              </div>
              <div class="form-actions">
                <button class="secondary-button" type="button" @click="showKnowledgeForm = false">取消</button>
                <button class="primary-button" type="submit">
                  <UploadCloud />
                  <span>上传</span>
                </button>
              </div>
            </form>

            <form v-if="resourceTab === 'prompts' && showPromptForm" class="resource-form" @submit.prevent="submitPromptAsset">
              <div class="form-grid resource-form-grid">
                <label>
                  <span>提示词名称</span>
                  <input v-model="promptForm.name" required placeholder="运维问答系统提示词" />
                </label>
                <label>
                  <span>提示词类型</span>
                  <select v-model="promptForm.prompt_type">
                    <option value="system">系统提示词</option>
                    <option value="instruction">任务指令</option>
                    <option value="template">模板</option>
                    <option value="other">其他</option>
                  </select>
                </label>
                <label>
                  <span>描述</span>
                  <input v-model="promptForm.description" placeholder="默认回答约束" />
                </label>
                <label class="upload-field">
                  <input id="prompt-file" type="file" accept=".txt" @change="onPromptFileChange" />
                  <FileUp />
                  <span>{{ promptFileName }}</span>
                </label>
              </div>
              <label class="prompt-content-field">
                <span>提示词内容</span>
                <textarea v-model="promptForm.content" maxlength="20000" placeholder="请输入提示词内容"></textarea>
              </label>
              <div class="form-actions">
                <button class="secondary-button" type="button" @click="showPromptForm = false">取消</button>
                <button class="primary-button" type="submit">
                  <UploadCloud />
                  <span>上传</span>
                </button>
              </div>
            </form>

            <div v-if="resourceTab === 'knowledge'" class="resource-table-block">
              <div class="resource-batchbar">
                <span>已选 {{ selectedKnowledgeIds.length }} 项</span>
                <button class="secondary-button danger-text" type="button" @click="batchDeleteKnowledgeSources">
                  <Trash2 />
                  <span>批量删除</span>
                </button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th class="select-col">
                        <input class="row-check" type="checkbox" :checked="allFilteredKnowledgeSelected" @change="setAllKnowledgeSelection" />
                      </th>
                      <th>知识源</th>
                      <th>类型</th>
                      <th>文件</th>
                      <th>编辑时间</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!filteredKnowledgeSources.length">
                      <td colspan="6" class="empty">暂无知识源</td>
                    </tr>
                    <tr v-for="item in filteredKnowledgeSources" :key="item.id">
                      <td class="select-col">
                        <input
                          class="row-check"
                          type="checkbox"
                          :checked="selectedKnowledgeIds.includes(item.id)"
                          @change="toggleKnowledgeSelection(item.id)"
                        />
                      </td>
                      <td>
                        <strong>{{ item.name }}</strong>
                        <small>{{ item.description || item.preview || "-" }}</small>
                      </td>
                      <td><span class="tag processing">{{ item.type_label }}</span></td>
                      <td>
                        <span>{{ item.original_name || item.stored_name }}</span>
                        <small>{{ statsText(item) }} · {{ formatSize(item.size_bytes) }}</small>
                      </td>
                      <td>{{ formatDate(item.updated_at) }}</td>
                      <td class="action-col">
                        <button class="text-button danger-text" type="button" @click="deleteKnowledgeSource(item)">
                          <Trash2 />
                          <span>删除</span>
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div v-if="resourceTab === 'prompts'" class="resource-table-block">
              <div class="resource-batchbar">
                <span>已选 {{ selectedPromptIds.length }} 项</span>
                <button class="secondary-button danger-text" type="button" @click="batchDeletePromptAssets">
                  <Trash2 />
                  <span>批量删除</span>
                </button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th class="select-col">
                        <input class="row-check" type="checkbox" :checked="allFilteredPromptSelected" @change="setAllPromptSelection" />
                      </th>
                      <th>提示词</th>
                      <th>类型</th>
                      <th>内容摘要</th>
                      <th>编辑时间</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!filteredPromptAssets.length">
                      <td colspan="6" class="empty">暂无提示词</td>
                    </tr>
                    <tr v-for="item in filteredPromptAssets" :key="item.id">
                      <td class="select-col">
                        <input
                          class="row-check"
                          type="checkbox"
                          :checked="selectedPromptIds.includes(item.id)"
                          @change="togglePromptSelection(item.id)"
                        />
                      </td>
                      <td>
                        <strong>{{ item.name }}</strong>
                        <small>{{ item.description || item.original_name || "-" }}</small>
                      </td>
                      <td><span class="tag">{{ item.type_label }}</span></td>
                      <td>
                        <span>{{ item.preview || "-" }}</span>
                        <small>{{ formatSize(item.size_bytes) }}</small>
                      </td>
                      <td>{{ formatDate(item.updated_at) }}</td>
                      <td class="action-col">
                        <button class="text-button danger-text" type="button" @click="deletePromptAsset(item)">
                          <Trash2 />
                          <span>删除</span>
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </section>

        <section v-if="currentPage === 'agentManage'" class="agent-manage-page">
          <div class="agent-manage-grid">
            <section class="panel-card">
              <div class="panel-title">
                <h2>智能体列表</h2>
                <button class="secondary-button" type="button" @click="refreshAgentManagement"><RefreshCw />刷新</button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>智能体名称</th>
                      <th>节点数</th>
                      <th>状态</th>
                      <th>接口服务</th>
                      <th>更新时间</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!agentWorkflows.length">
                      <td colspan="6" class="empty">暂无智能体</td>
                    </tr>
                    <tr v-for="agent in agentWorkflows" :key="agent.id" :class="{ selected: managedAgentId === agent.id }">
                      <td>
                        <strong>{{ agent.name }}</strong>
                        <small>{{ agent.description || "顺序执行链路" }}</small>
                      </td>
                      <td>{{ agent.node_count || agent.nodes?.length || 0 }}</td>
                      <td><span class="tag success">{{ statusText(agent.status) }}</span></td>
                      <td>
                        <label class="switch-line">
                          <input type="checkbox" :checked="Boolean(agentServiceFor(agent))" @change="toggleAgentService(agent, $event)" />
                          <span>{{ agentServiceFor(agent) ? "运行中" : "未开启" }}</span>
                        </label>
                        <small v-if="agentServiceFor(agent)" class="service-url">{{ agentServiceFor(agent).url }}</small>
                      </td>
                      <td>{{ formatDate(agent.updated_at) }}</td>
                      <td class="action-col agent-action-col">
                        <button class="text-button" type="button" @click="loadAgentToBuilder(agent)">编辑</button>
                        <button class="text-button" type="button" @click="managedAgentId = agent.id">测试</button>
                        <button class="text-button danger-text" type="button" @click="deleteAgentWorkflow(agent)">删除</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section class="panel-card">
              <div class="panel-title">
                <h2>智能体测试</h2>
                <span class="tag">{{ managedAgent?.name || "未选择" }}</span>
              </div>
              <div class="agent-test-panel">
                <label>
                  <span>测试输入</span>
                  <textarea v-model="managedAgentTestInput" placeholder="输入一段用户问题"></textarea>
                </label>
                <button class="primary-button" type="button" :disabled="managedAgentRunning" @click="runManagedAgent">
                  <Play />
                  <span>{{ managedAgentRunning ? "运行中" : "测试运行" }}</span>
                </button>
                <pre>{{ managedAgentTestResult?.output || "选择一个智能体后可测试运行" }}</pre>
                <div v-if="managedAgentTestResult?.trace?.length" class="trace-list">
                  <strong>执行轨迹</strong>
                  <div v-for="item in managedAgentTestResult.trace" :key="`${item.index}-${item.node}`">
                    <span>{{ item.index }}. {{ item.node }}</span>
                    <small>{{ item.output }}</small>
                  </div>
                </div>
              </div>
            </section>

            <section class="panel-card service-management-panel">
              <div class="panel-title">
                <h2>服务管理</h2>
                <button class="secondary-button" type="button" @click="loadAgentServices"><RefreshCw />刷新</button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>服务智能体</th>
                      <th>服务 URL</th>
                      <th>调用次数</th>
                      <th>最近调用</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!runningAgentServices.length">
                      <td colspan="5" class="empty">暂无运行中的智能体服务</td>
                    </tr>
                    <tr v-for="service in runningAgentServices" :key="service.id">
                      <td>
                        <strong>{{ service.agent_name }}</strong>
                        <small>{{ service.agent_description || `${service.node_count || 0} 个节点` }}</small>
                      </td>
                      <td>
                        <code class="service-url">{{ service.url }}</code>
                        <small>POST JSON: {"input_text": "用户问题"}</small>
                      </td>
                      <td>{{ service.invoke_count || 0 }}</td>
                      <td>{{ service.last_invoked_at ? formatDate(service.last_invoked_at) : "-" }}</td>
                      <td class="action-col agent-action-col">
                        <button class="text-button" type="button" @click="copyServiceUrl(service)">
                          <Copy />
                          <span>复制</span>
                        </button>
                        <button class="text-button danger-text" type="button" @click="stopAgentService(service)">
                          <Power />
                          <span>停止</span>
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
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
