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
  Plus,
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
  graphCreate: { title: "新建知识库", tab: "知识图谱", subtitle: "从文档、表格、图片中抽取实体关系，构建多层次知识图谱" },
  graphFusion: { title: "图谱融合构建", tab: "知识图谱", subtitle: "从已有知识图谱中进行实体对齐、关系归并和溯源融合" },
  graphDatabase: { title: "数据库导出构建", tab: "知识图谱", subtitle: "从业务数据库抽取主题数据并映射生成新的知识图谱" },
  graphVersions: { title: "知识图谱版本管理", tab: "知识图谱", subtitle: "按知识库查看图谱版本，并展示节点与关系的可视化链路" },
};

const currentPage = ref("train");
const baseModels = ref([]);
const downloadJobs = ref([]);
const trainingJobs = ref([]);
const fineTunedModels = ref([]);
const knowledgeSources = ref([]);
const promptAssets = ref([]);
const agentWorkflows = ref([]);
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
const selectedFusionGraphIds = ref(["kg-equip-compressor", "kg-process-compressor"]);
const activeFusionLibraryId = ref("kg-equip");
const fusionSearch = ref("");
const selectedWorkflowNodeId = ref("input-node");
const workflowTestInput = ref("");
const workflowTestResult = ref(null);
const workflowRunning = ref(false);
const managedAgentId = ref("");
const managedAgentTestInput = ref("");
const managedAgentTestResult = ref(null);
const managedAgentRunning = ref(false);
const createSourceFiles = ref([]);
const createImageFiles = ref([]);
const graphAnalysisTaskId = ref("");
const graphAnalysisTasks = ref([]);
const selectedGraphLibraryId = ref("kg-equip");
const selectedGraphVersionId = ref("v3");
const graphBuilderLoaded = ref(false);
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

const graphCreateForm = reactive({
  name: "压缩机售后知识图谱",
  businessDomain: "工业设备运维",
  layerStrategy: "文档层 + 实体层 + 关系层",
  description: "覆盖设备结构、故障现象、备件编码和工艺步骤的统一图谱。",
});

const graphRawHeaders = ref([
  { id: "header-1", name: "设备名称", sample: "压缩机1号", role: "主实体列", description: "每行对应一台设备对象" },
  { id: "header-2", name: "设备编码", sample: "COMP-001", role: "标识列", description: "可作为实体唯一标识" },
  { id: "header-3", name: "故障现象", sample: "高温告警", role: "事件列", description: "描述设备故障或异常现象" },
  { id: "header-4", name: "故障原因", sample: "润滑不足", role: "原因列", description: "可映射为导致故障的原因节点" },
  { id: "header-5", name: "处理措施", sample: "检查油路", role: "动作列", description: "可映射为处置动作节点" },
  { id: "header-6", name: "责任班组", sample: "维保二班", role: "组织列", description: "可映射为组织或责任主体" },
  { id: "header-7", name: "备件编码", sample: "6208-ZZ", role: "物料列", description: "可映射为备件或物料节点" },
  { id: "header-8", name: "安装位置", sample: "一车间", role: "位置列", description: "可映射为位置节点" },
]);

const graphSampleRows = ref([
  {
    "设备名称": "压缩机1号",
    "设备编码": "COMP-001",
    "故障现象": "高温告警",
    "故障原因": "润滑不足",
    "处理措施": "检查油路",
    "责任班组": "维保二班",
    "备件编码": "6208-ZZ",
    "安装位置": "一车间",
  },
  {
    "设备名称": "压缩机2号",
    "设备编码": "COMP-002",
    "故障现象": "异响",
    "故障原因": "轴承磨损",
    "处理措施": "更换主轴轴承",
    "责任班组": "维保一班",
    "备件编码": "BR-6208-A",
    "安装位置": "二车间",
  },
]);

const graphModelNodeSuggestions = ref([
  { id: "node-s-1", header: "设备名称", node_type: "设备", confidence: "高", reason: "字段值稳定，语义明确指向设备实体" },
  { id: "node-s-2", header: "故障现象", node_type: "故障", confidence: "高", reason: "字段内容为异常事件或故障表现" },
  { id: "node-s-3", header: "故障原因", node_type: "故障", confidence: "中", reason: "字段内容更适合作为原因类故障节点" },
  { id: "node-s-4", header: "处理措施", node_type: "工艺步骤", confidence: "中", reason: "字段描述操作动作，适合映射为处置步骤" },
  { id: "node-s-5", header: "责任班组", node_type: "组织", confidence: "高", reason: "字段值体现责任主体与组织对象" },
  { id: "node-s-6", header: "备件编码", node_type: "备件", confidence: "高", reason: "字段值为备件或物料编码" },
  { id: "node-s-7", header: "安装位置", node_type: "位置", confidence: "高", reason: "字段值可映射为位置类节点" },
]);

const graphModelRelationSuggestions = ref([
  { id: "rel-s-1", source_header: "设备名称", relation: "出现", target_header: "故障现象", confidence: "高", reason: "设备与故障现象存在直接挂接关系" },
  { id: "rel-s-2", source_header: "故障原因", relation: "导致", target_header: "故障现象", confidence: "高", reason: "原因字段与现象字段形成因果关系" },
  { id: "rel-s-3", source_header: "处理措施", relation: "处理", target_header: "故障现象", confidence: "中", reason: "措施字段通常用于处置故障现象" },
  { id: "rel-s-4", source_header: "责任班组", relation: "负责", target_header: "设备名称", confidence: "高", reason: "责任主体与设备对象形成责任关系" },
  { id: "rel-s-5", source_header: "设备名称", relation: "使用", target_header: "备件编码", confidence: "中", reason: "设备与备件存在关联使用或替换关系" },
  { id: "rel-s-6", source_header: "设备名称", relation: "位于", target_header: "安装位置", confidence: "高", reason: "设备字段与安装位置字段形成空间关系" },
]);

const graphHeaderConfirmations = ref([
  { id: "confirm-1", source_header: "设备名称", source_type: "设备", relation: "出现", target_header: "故障现象", target_type: "故障", status: "待确认" },
  { id: "confirm-2", source_header: "故障原因", source_type: "故障", relation: "导致", target_header: "故障现象", target_type: "故障", status: "待确认" },
  { id: "confirm-3", source_header: "处理措施", source_type: "工艺步骤", relation: "处理", target_header: "故障现象", target_type: "故障", status: "待确认" },
  { id: "confirm-4", source_header: "责任班组", source_type: "组织", relation: "负责", target_header: "设备名称", target_type: "设备", status: "待确认" },
]);
const graphTriplePreviewItems = ref([]);

const graphNodeTypeForm = reactive({
  name: "",
  code: "",
  description: "",
});

const graphNodeTypes = ref([
  { id: "node-equipment", name: "设备", code: "equipment", description: "压缩机、冷却器、油泵等设备对象", count: 42 },
  { id: "node-part", name: "部件", code: "part", description: "轴承、转子、油路、壳体等组成部件", count: 86 },
  { id: "node-fault", name: "故障", code: "fault", description: "高温告警、润滑不足、异响等异常现象", count: 28 },
  { id: "node-process", name: "工艺步骤", code: "process_step", description: "装配、校验、保养、巡检等业务步骤", count: 35 },
  { id: "node-spare", name: "备件", code: "spare_part", description: "备件编码、替代件和物料对象", count: 19 },
]);

const graphRelationTypeForm = reactive({
  name: "",
  code: "",
  description: "",
  direction: "单向",
});

const graphRelationTypes = ref([
  { id: "rel-contains", name: "包含", code: "contains", description: "用于表示设备与部件、总成与子件的组成关系", direction: "单向" },
  { id: "rel-causes", name: "导致", code: "causes", description: "用于表示故障原因和故障现象、事件后果关系", direction: "单向" },
  { id: "rel-uses", name: "使用", code: "uses", description: "用于表示工艺步骤对工具、备件、文档的使用关系", direction: "单向" },
  { id: "rel-responsible", name: "负责", code: "responsible_for", description: "用于表示组织或角色对设备、任务的责任关系", direction: "单向" },
]);

const graphRelationRuleForm = reactive({
  source_type: "设备",
  relation: "包含",
  target_type: "部件",
  example: "",
});

const graphTripleForm = reactive({
  source: "",
  source_type: "设备",
  relation: "包含",
  target: "",
  target_type: "部件",
  origin: "手动新增",
});

const graphRelationRules = ref([
  { id: "rule-1", source_type: "设备", relation: "包含", target_type: "部件", example: "压缩机 -> 包含 -> 轴承" },
  { id: "rule-2", source_type: "故障", relation: "导致", target_type: "故障", example: "润滑不足 -> 导致 -> 高温告警" },
  { id: "rule-3", source_type: "工艺步骤", relation: "使用", target_type: "备件", example: "转子装配 -> 使用 -> 主轴轴承" },
  { id: "rule-4", source_type: "组织", relation: "负责", target_type: "设备", example: "维保二班 -> 负责 -> 1#压缩机" },
]);

const graphExtractedTriples = ref([
  {
    id: "triple-1",
    source: "压缩机",
    source_type: "设备",
    relation: "包含",
    target: "轴承",
    target_type: "部件",
    origin: "测试数据_3.xlsx",
    status: "待确认",
  },
  {
    id: "triple-2",
    source: "润滑不足",
    source_type: "故障",
    relation: "导致",
    target: "高温告警",
    target_type: "故障",
    origin: "测试数据_3.xlsx",
    status: "已确认",
  },
  {
    id: "triple-3",
    source: "转子装配",
    source_type: "工艺步骤",
    relation: "使用",
    target: "主轴轴承",
    target_type: "备件",
    origin: "测试数据_3.xlsx",
    status: "待确认",
  },
]);

const graphFusionForm = reactive({
  name: "压缩机综合知识图谱",
  fusionMode: "实体对齐优先",
  conflictRule: "保留高可信版本并生成溯源链路",
  description: "融合设备台账图谱、工艺图谱与运维案例图谱。",
});

const graphDatabaseForm = reactive({
  name: "设备主数据图谱",
  databaseType: "PostgreSQL",
  connectionName: "factory-mes-readonly",
  schemaName: "public",
  tableNames: "equipment_master, maintenance_order, spare_parts",
  syncMode: "按主题导出后构图",
});

const graphAlignmentPreview = ref([
  {
    id: "align-1",
    left: "设备维护知识库 / 压缩机维护子库 / 压缩机机组",
    right: "装配工艺知识库 / 压缩机装配子库 / 压缩机总成",
    confidence: "高",
    rule: "同义词 + 编码相似度",
    decision: "自动合并",
  },
  {
    id: "align-2",
    left: "设备主数据知识库 / 设备台账子库 / 1#压缩机",
    right: "设备维护知识库 / 压缩机维护子库 / 压缩机1号",
    confidence: "高",
    rule: "设备编码一致",
    decision: "自动合并",
  },
  {
    id: "align-3",
    left: "装配工艺知识库 / 质检工艺子库 / 力矩校验",
    right: "设备维护知识库 / 压缩机维护子库 / 螺栓力矩检查",
    confidence: "中",
    rule: "语义相似",
    decision: "待确认",
  },
]);

const graphConflictItems = ref([
  {
    id: "conflict-1",
    field: "责任班组",
    entity: "1#压缩机",
    left: "总装一班",
    right: "维保二班",
    severity: "高",
    suggestion: "保留高可信版本并记录来源",
  },
  {
    id: "conflict-2",
    field: "备件编码",
    entity: "主轴轴承",
    left: "BR-6208-A",
    right: "6208-ZZ",
    severity: "中",
    suggestion: "人工确认后建立替代件关系",
  },
  {
    id: "conflict-3",
    field: "工艺步骤名称",
    entity: "转子校准",
    left: "动平衡校验",
    right: "转子平衡检测",
    severity: "中",
    suggestion: "并存保留并增加同义关系",
  },
]);


const graphLibraries = ref([
  {
    id: "kg-equip",
    name: "设备维护知识库",
    domain: "设备运维",
    source: "文档 / 表格 / 图片",
    layers: ["文档层", "实体层", "关系层", "故障事件层"],
    entity_count: 326,
    relation_count: 892,
    updated_at: "2026-05-30T09:30:00+08:00",
    owner: "知识工程组",
    children: [
      {
        id: "kg-equip-compressor",
        name: "压缩机维护子库",
        scene: "压缩机售后与巡检",
        entity_count: 138,
        relation_count: 326,
        updated_at: "2026-05-30T09:30:00+08:00",
      },
      {
        id: "kg-equip-cooling",
        name: "冷却系统子库",
        scene: "冷却回路故障处理",
        entity_count: 92,
        relation_count: 210,
        updated_at: "2026-05-28T16:40:00+08:00",
      },
      {
        id: "kg-equip-lube",
        name: "润滑系统子库",
        scene: "润滑异常与保养策略",
        entity_count: 96,
        relation_count: 244,
        updated_at: "2026-05-27T11:20:00+08:00",
      },
    ],
    versions: [
      {
        id: "v3",
        label: "V3.2 当前生产版",
        status: "ready",
        updated_at: "2026-05-30T09:30:00+08:00",
        summary: "补充压缩机故障树和典型维修工序，新增 74 个关系边。",
        metrics: { entities: 326, relations: 892, sources: 18 },
        nodes: [
          { id: "n1", label: "压缩机", type: "核心设备" },
          { id: "n2", label: "高温告警", type: "故障现象" },
          { id: "n3", label: "润滑不足", type: "原因" },
          { id: "n4", label: "检查油路", type: "处置动作" },
        ],
        edges: [
          "压缩机 -> 高温告警",
          "高温告警 -> 润滑不足",
          "润滑不足 -> 检查油路",
        ],
      },
      {
        id: "v2",
        label: "V3.1 上一版本",
        status: "succeeded",
        updated_at: "2026-05-18T14:12:00+08:00",
        summary: "完成备件编码映射，补充 2 个图片识别规则。",
        metrics: { entities: 298, relations: 818, sources: 15 },
        nodes: [
          { id: "n1", label: "压缩机", type: "核心设备" },
          { id: "n2", label: "轴承", type: "部件" },
          { id: "n3", label: "备件编码", type: "主数据" },
        ],
        edges: [
          "压缩机 -> 轴承",
          "轴承 -> 备件编码",
        ],
      },
    ],
  },
  {
    id: "kg-process",
    name: "装配工艺知识库",
    domain: "制造工艺",
    source: "已有知识图谱融合",
    layers: ["工艺层", "工步层", "质检层"],
    entity_count: 214,
    relation_count: 541,
    updated_at: "2026-05-25T16:05:00+08:00",
    owner: "工艺信息化组",
    children: [
      {
        id: "kg-process-compressor",
        name: "压缩机装配子库",
        scene: "总装、校验与交付工步",
        entity_count: 104,
        relation_count: 248,
        updated_at: "2026-05-25T16:05:00+08:00",
      },
      {
        id: "kg-process-rotor",
        name: "转子装配子库",
        scene: "转子、轴承与动平衡工艺",
        entity_count: 66,
        relation_count: 171,
        updated_at: "2026-05-20T09:50:00+08:00",
      },
      {
        id: "kg-process-quality",
        name: "质检工艺子库",
        scene: "力矩、间隙和质检卡映射",
        entity_count: 44,
        relation_count: 122,
        updated_at: "2026-05-18T15:30:00+08:00",
      },
    ],
    versions: [
      {
        id: "v4",
        label: "V2.0 当前生产版",
        status: "ready",
        updated_at: "2026-05-25T16:05:00+08:00",
        summary: "将装配工步与质检卡对齐，增加工艺变更溯源节点。",
        metrics: { entities: 214, relations: 541, sources: 9 },
        nodes: [
          { id: "n1", label: "装配工序", type: "工艺节点" },
          { id: "n2", label: "力矩校验", type: "质检项" },
          { id: "n3", label: "工步卡", type: "文档" },
        ],
        edges: [
          "装配工序 -> 力矩校验",
          "工步卡 -> 装配工序",
        ],
      },
    ],
  },
  {
    id: "kg-master",
    name: "设备主数据知识库",
    domain: "主数据治理",
    source: "数据库导出",
    layers: ["主数据层", "组织层", "运维对象层"],
    entity_count: 482,
    relation_count: 1260,
    updated_at: "2026-05-29T11:10:00+08:00",
    owner: "数据治理组",
    children: [
      {
        id: "kg-master-equipment",
        name: "设备台账子库",
        scene: "设备编码、位置和责任班组",
        entity_count: 188,
        relation_count: 486,
        updated_at: "2026-05-29T11:10:00+08:00",
      },
      {
        id: "kg-master-spare",
        name: "备件主数据子库",
        scene: "备件编码、替代件和物料分类",
        entity_count: 160,
        relation_count: 404,
        updated_at: "2026-05-27T10:45:00+08:00",
      },
      {
        id: "kg-master-order",
        name: "工单主题子库",
        scene: "维保工单、故障标签与执行记录",
        entity_count: 134,
        relation_count: 370,
        updated_at: "2026-05-24T14:10:00+08:00",
      },
    ],
    versions: [
      {
        id: "v5",
        label: "V1.6 当前生产版",
        status: "ready",
        updated_at: "2026-05-29T11:10:00+08:00",
        summary: "从 MES / EAM 同步设备、位置、责任班组信息并生成主数据图谱。",
        metrics: { entities: 482, relations: 1260, sources: 6 },
        nodes: [
          { id: "n1", label: "设备编码", type: "主键" },
          { id: "n2", label: "安装位置", type: "位置" },
          { id: "n3", label: "责任班组", type: "组织" },
          { id: "n4", label: "保养策略", type: "策略" },
        ],
        edges: [
          "设备编码 -> 安装位置",
          "设备编码 -> 责任班组",
          "设备编码 -> 保养策略",
        ],
      },
    ],
  },
]);

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
const promptFileName = computed(() => promptFile.value?.name || "可选：上传提示词文件");
const readyBaseModels = computed(() => baseModels.value.filter((model) => model.status === "ready"));
const selectedJob = computed(() => trainingJobs.value.find((job) => job.id === selectedJobId.value));
const latestDownloadJobs = computed(() => downloadJobs.value.slice(0, 4));
const latestTrainingJobs = computed(() => trainingJobs.value.slice(0, 5));
const ragModels = computed(() => fineTunedModels.value.filter((model) => model.model_type === "rag"));
const trainableFineTunedModels = computed(() => fineTunedModels.value);
const latestRagModels = computed(() => ragModels.value.slice(0, 5));
const filteredKnowledgeSources = computed(() => filterResources(knowledgeSources.value));
const filteredPromptAssets = computed(() => filterResources(promptAssets.value));
const graphHeaderOptions = computed(() => graphRawHeaders.value.map((item) => item.name));
const graphLibraryCount = computed(() => graphLibraries.value.length);
const graphVersionCount = computed(() => graphLibraries.value.reduce((sum, item) => sum + item.versions.length, 0));
const graphEntityCount = computed(() => graphLibraries.value.reduce((sum, item) => sum + item.entity_count, 0));
const graphRelationCount = computed(() => graphLibraries.value.reduce((sum, item) => sum + item.relation_count, 0));
const activeFusionLibrary = computed(() => graphLibraries.value.find((item) => item.id === activeFusionLibraryId.value) || graphLibraries.value[0]);
const activeFusionChildren = computed(() => {
  const keyword = fusionSearch.value.trim().toLowerCase();
  const items = activeFusionLibrary.value?.children || [];
  if (!keyword) return items;
  return items.filter((child) => [child.name, child.scene].some((value) => String(value).toLowerCase().includes(keyword)));
});
const fusionSourceGraphs = computed(() =>
  graphLibraries.value.flatMap((item) =>
    (item.children || [])
      .filter((child) => selectedFusionGraphIds.value.includes(child.id))
      .map((child) => ({ ...child, parent_name: item.name })),
  ),
);
const graphAlignmentPendingCount = computed(() => graphAlignmentPreview.value.filter((item) => item.decision === "待确认").length);
const graphConflictHighCount = computed(() => graphConflictItems.value.filter((item) => item.severity === "高").length);
const currentGraphLibrary = computed(() => graphLibraries.value.find((item) => item.id === selectedGraphLibraryId.value) || graphLibraries.value[0]);
const currentGraphVersion = computed(
  () => currentGraphLibrary.value?.versions.find((item) => item.id === selectedGraphVersionId.value) || currentGraphLibrary.value?.versions[0],
);
const latestGraphLibraries = computed(() => [...graphLibraries.value].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)).slice(0, 4));
const selectedWorkflowNode = computed(() => workflowForm.nodes.find((node) => node.id === selectedWorkflowNodeId.value));
const managedAgent = computed(() => agentWorkflows.value.find((agent) => agent.id === managedAgentId.value));
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
  const rawText = await response.text();
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = rawText ? JSON.parse(rawText) : {};
      detail = formatErrorDetail(payload.detail) || detail;
    } catch {
      detail = rawText || detail;
    }
    throw new Error(detail);
  }
  return rawText ? JSON.parse(rawText) : {};
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

function isGraphPage(page) {
  return ["graphCreate", "graphFusion", "graphDatabase", "graphVersions"].includes(page);
}

function switchPage(page) {
  currentPage.value = page;
  if (isGraphPage(page) && !graphBuilderLoaded.value) {
    loadGraphBuilderConfig().catch((error) => showToast(error.message));
  }
}

function onCreateSourceFilesChange(event) {
  createSourceFiles.value = Array.from(event.target.files || []);
  const firstFile = createSourceFiles.value[0];
  if (firstFile) {
    analyzeGraphTable(firstFile).catch((error) => showToast(error.message));
  }
}

function onCreateImageFilesChange(event) {
  createImageFiles.value = Array.from(event.target.files || []);
}

async function analyzeGraphTable(file) {
  const form = new FormData();
  form.append("table_file", file);
  form.append("knowledge_base_name", graphCreateForm.name);
  const payload = await fetchJSON("/api/graph-builder/analyze-table", {
    method: "POST",
    body: form,
  });
  if (Array.isArray(payload.headers) && payload.headers.length) {
    graphRawHeaders.value = payload.headers;
  }
  if (Array.isArray(payload.sample_rows)) {
    graphSampleRows.value = payload.sample_rows;
  }
  if (Array.isArray(payload.node_suggestions)) {
    graphModelNodeSuggestions.value = payload.node_suggestions;
  }
  if (Array.isArray(payload.relation_suggestions)) {
    graphModelRelationSuggestions.value = payload.relation_suggestions;
  }
  if (Array.isArray(payload.confirmation_rules) && payload.confirmation_rules.length) {
    graphHeaderConfirmations.value = payload.confirmation_rules;
  }
  if (payload.task?.id) {
    graphAnalysisTaskId.value = payload.task.id;
    await loadGraphAnalysisTasks();
    await loadGraphTriplePreview(payload.task.id);
  }
  showToast(payload.analysis_source === "llm" ? "已完成大模型表头理解与关系建议" : "已完成表头解析与关系建议");
}

async function loadGraphAnalysisTasks() {
  const payload = await fetchJSON("/api/graph-builder/analysis-tasks");
  graphAnalysisTasks.value = payload.items || [];
}

async function loadGraphTriplePreview(taskId = graphAnalysisTaskId.value) {
  if (!taskId) {
    graphTriplePreviewItems.value = [];
    return;
  }
  const payload = await fetchJSON(`/api/graph-builder/analysis-tasks/${taskId}/triples-preview`);
  graphTriplePreviewItems.value = payload.items || [];
}

async function loadGraphAnalysisTask(taskId) {
  const payload = await fetchJSON(`/api/graph-builder/analysis-tasks/${taskId}`);
  const task = payload.item;
  graphAnalysisTaskId.value = task.id;
  graphRawHeaders.value = task.headers || [];
  graphSampleRows.value = task.sample_rows || [];
  graphModelNodeSuggestions.value = task.node_suggestions || [];
  graphModelRelationSuggestions.value = task.relation_suggestions || [];
  graphHeaderConfirmations.value = task.confirmation_rules || [];
  await loadGraphTriplePreview(task.id);
}

async function persistGraphHeaderConfirmations() {
  if (!graphAnalysisTaskId.value) return;
  await fetchJSON(`/api/graph-builder/analysis-tasks/${graphAnalysisTaskId.value}/confirmation-rules`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: graphHeaderConfirmations.value }),
  });
  await loadGraphAnalysisTasks();
  await loadGraphTriplePreview(graphAnalysisTaskId.value);
}

async function persistGraphBuilderConfig() {
  await Promise.all([
    fetchJSON("/api/graph-builder/node-types", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: graphNodeTypes.value }),
    }),
    fetchJSON("/api/graph-builder/relation-types", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: graphRelationTypes.value }),
    }),
    fetchJSON("/api/graph-builder/relation-rules", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: graphRelationRules.value }),
    }),
    fetchJSON("/api/graph-builder/triples", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: graphExtractedTriples.value }),
    }),
  ]);
}

async function loadGraphBuilderConfig() {
  const [nodeTypesPayload, relationTypesPayload, relationRulesPayload, triplesPayload] = await Promise.all([
    fetchJSON("/api/graph-builder/node-types"),
    fetchJSON("/api/graph-builder/relation-types"),
    fetchJSON("/api/graph-builder/relation-rules"),
    fetchJSON("/api/graph-builder/triples"),
  ]);

  const hasSavedConfig =
    (nodeTypesPayload.items || []).length ||
    (relationTypesPayload.items || []).length ||
    (relationRulesPayload.items || []).length ||
    (triplesPayload.items || []).length;

  if ((nodeTypesPayload.items || []).length) graphNodeTypes.value = nodeTypesPayload.items;
  if ((relationTypesPayload.items || []).length) graphRelationTypes.value = relationTypesPayload.items;
  if ((relationRulesPayload.items || []).length) graphRelationRules.value = relationRulesPayload.items;
  if ((triplesPayload.items || []).length) graphExtractedTriples.value = triplesPayload.items;

  if (!hasSavedConfig) {
    await persistGraphBuilderConfig();
  }

  syncGraphBuilderForms();
  graphBuilderLoaded.value = true;
}

function syncGraphBuilderForms() {
  const firstNodeType = graphNodeTypes.value[0]?.name || "";
  const secondNodeType = graphNodeTypes.value[1]?.name || firstNodeType;
  const firstRelationType = graphRelationTypes.value[0]?.name || "";

  if (!graphRelationRuleForm.source_type || !graphNodeTypes.value.some((item) => item.name === graphRelationRuleForm.source_type)) {
    graphRelationRuleForm.source_type = firstNodeType;
  }
  if (!graphRelationRuleForm.target_type || !graphNodeTypes.value.some((item) => item.name === graphRelationRuleForm.target_type)) {
    graphRelationRuleForm.target_type = secondNodeType;
  }
  if (!graphRelationRuleForm.relation || !graphRelationTypes.value.some((item) => item.name === graphRelationRuleForm.relation)) {
    graphRelationRuleForm.relation = firstRelationType;
  }

  if (!graphTripleForm.source_type || !graphNodeTypes.value.some((item) => item.name === graphTripleForm.source_type)) {
    graphTripleForm.source_type = firstNodeType;
  }
  if (!graphTripleForm.target_type || !graphNodeTypes.value.some((item) => item.name === graphTripleForm.target_type)) {
    graphTripleForm.target_type = secondNodeType;
  }
  if (!graphTripleForm.relation || !graphRelationTypes.value.some((item) => item.name === graphTripleForm.relation)) {
    graphTripleForm.relation = firstRelationType;
  }
}

function submitGraphCreate() {
  if (!graphCreateForm.name.trim()) {
    showToast("请填写知识库名称");
    return;
  }
  if (!createSourceFiles.value.length && !createImageFiles.value.length) {
    showToast("请至少上传文档、表格或图片中的一种来源");
    return;
  }
  showToast("知识图谱构建任务已创建");
}

async function addGraphHeaderConfirmation() {
  graphHeaderConfirmations.value.push({
    id: `confirm-${Date.now()}`,
    source_header: graphHeaderOptions.value[0] || "",
    source_type: graphNodeTypes.value[0]?.name || "",
    relation: graphRelationTypes.value[0]?.name || "",
    target_header: graphHeaderOptions.value[1] || graphHeaderOptions.value[0] || "",
    target_type: graphNodeTypes.value[1]?.name || graphNodeTypes.value[0]?.name || "",
    status: "待确认",
  });
  await persistGraphHeaderConfirmations();
  showToast("已新增一条确认规则");
}

async function confirmGraphHeaderRule(id) {
  const item = graphHeaderConfirmations.value.find((rule) => rule.id === id);
  if (!item) return;
  item.status = "已确认";
  await persistGraphHeaderConfirmations();
  showToast("关系确认已更新");
}

async function deleteGraphHeaderRule(id) {
  graphHeaderConfirmations.value = graphHeaderConfirmations.value.filter((rule) => rule.id !== id);
  await persistGraphHeaderConfirmations();
  showToast("已删除一条确认规则");
}

async function addGraphNodeType() {
  if (!graphNodeTypeForm.name.trim() || !graphNodeTypeForm.code.trim()) {
    showToast("请填写节点类型名称和编码");
    return;
  }
  graphNodeTypes.value.unshift({
    id: `node-${Date.now()}`,
    name: graphNodeTypeForm.name.trim(),
    code: graphNodeTypeForm.code.trim(),
    description: graphNodeTypeForm.description.trim() || "手动新增节点类型",
    count: 0,
  });
  Object.assign(graphNodeTypeForm, { name: "", code: "", description: "" });
  syncGraphBuilderForms();
  await persistGraphBuilderConfig();
  showToast("节点类型已新增");
}

async function deleteGraphNodeType(id) {
  const node = graphNodeTypes.value.find((item) => item.id === id);
  graphNodeTypes.value = graphNodeTypes.value.filter((item) => item.id !== id);
  if (node) {
    graphRelationRules.value = graphRelationRules.value.filter(
      (item) => item.source_type !== node.name && item.target_type !== node.name,
    );
  }
  syncGraphBuilderForms();
  await persistGraphBuilderConfig();
  showToast("节点类型已删除");
}

async function addGraphRelationType() {
  if (!graphRelationTypeForm.name.trim() || !graphRelationTypeForm.code.trim()) {
    showToast("请填写关系类型名称和编码");
    return;
  }
  graphRelationTypes.value.unshift({
    id: `rel-${Date.now()}`,
    name: graphRelationTypeForm.name.trim(),
    code: graphRelationTypeForm.code.trim(),
    description: graphRelationTypeForm.description.trim() || "手动新增关系类型",
    direction: graphRelationTypeForm.direction,
  });
  Object.assign(graphRelationTypeForm, { name: "", code: "", description: "", direction: "单向" });
  syncGraphBuilderForms();
  await persistGraphBuilderConfig();
  showToast("关系类型已新增");
}

async function deleteGraphRelationType(id) {
  const relation = graphRelationTypes.value.find((item) => item.id === id);
  graphRelationTypes.value = graphRelationTypes.value.filter((item) => item.id !== id);
  if (relation) {
    graphRelationRules.value = graphRelationRules.value.filter((item) => item.relation !== relation.name);
    const fallbackRelation = graphRelationTypes.value[0]?.name || "";
    graphExtractedTriples.value = graphExtractedTriples.value.map((item) =>
      item.relation === relation.name ? { ...item, relation: fallbackRelation, status: "待确认" } : item,
    );
  }
  syncGraphBuilderForms();
  await persistGraphBuilderConfig();
  showToast("关系类型已删除");
}

async function addGraphRelationRule() {
  if (!graphRelationRuleForm.source_type || !graphRelationRuleForm.relation || !graphRelationRuleForm.target_type) {
    showToast("请补充关系约束信息");
    return;
  }
  graphRelationRules.value.unshift({
    id: `rule-${Date.now()}`,
    source_type: graphRelationRuleForm.source_type,
    relation: graphRelationRuleForm.relation,
    target_type: graphRelationRuleForm.target_type,
    example:
      graphRelationRuleForm.example.trim() ||
      `${graphRelationRuleForm.source_type} -> ${graphRelationRuleForm.relation} -> ${graphRelationRuleForm.target_type}`,
  });
  Object.assign(graphRelationRuleForm, {
    source_type: graphNodeTypes.value[0]?.name || "",
    relation: graphRelationTypes.value[0]?.name || "",
    target_type: graphNodeTypes.value[1]?.name || graphNodeTypes.value[0]?.name || "",
    example: "",
  });
  syncGraphBuilderForms();
  await persistGraphBuilderConfig();
  showToast("关系约束已新增");
}

async function deleteGraphRelationRule(id) {
  graphRelationRules.value = graphRelationRules.value.filter((item) => item.id !== id);
  await persistGraphBuilderConfig();
  showToast("关系约束已删除");
}

async function addGraphTriple() {
  if (!graphTripleForm.source.trim() || !graphTripleForm.target.trim()) {
    showToast("请填写起点节点和终点节点");
    return;
  }
  graphExtractedTriples.value.unshift({
    id: `triple-${Date.now()}`,
    source: graphTripleForm.source.trim(),
    source_type: graphTripleForm.source_type,
    relation: graphTripleForm.relation,
    target: graphTripleForm.target.trim(),
    target_type: graphTripleForm.target_type,
    origin: graphTripleForm.origin.trim() || "手动新增",
    status: "待确认",
  });
  Object.assign(graphTripleForm, {
    source: "",
    source_type: graphNodeTypes.value[0]?.name || "",
    relation: graphRelationTypes.value[0]?.name || "",
    target: "",
    target_type: graphNodeTypes.value[1]?.name || graphNodeTypes.value[0]?.name || "",
    origin: "手动新增",
  });
  await persistGraphBuilderConfig();
  showToast("三元组已新增");
}

async function confirmGraphTriple(id) {
  const item = graphExtractedTriples.value.find((triple) => triple.id === id);
  if (!item) return;
  item.status = "已确认";
  await persistGraphBuilderConfig();
  showToast("三元组已确认");
}

async function ignoreGraphTriple(id) {
  const item = graphExtractedTriples.value.find((triple) => triple.id === id);
  if (!item) return;
  item.status = "已忽略";
  await persistGraphBuilderConfig();
  showToast("三元组已忽略");
}

async function updateGraphTripleRelation(id, relation) {
  const item = graphExtractedTriples.value.find((triple) => triple.id === id);
  if (!item) return;
  item.relation = relation;
  item.status = "待确认";
  await persistGraphBuilderConfig();
  showToast("三元组关系已更新");
}

function setActiveFusionLibrary(id) {
  activeFusionLibraryId.value = id;
}

function toggleFusionGraph(id) {
  selectedFusionGraphIds.value = selectedFusionGraphIds.value.includes(id)
    ? selectedFusionGraphIds.value.filter((itemId) => itemId !== id)
    : [...selectedFusionGraphIds.value, id];
}

function submitGraphFusion() {
  if (selectedFusionGraphIds.value.length < 2) {
    showToast("请至少选择两个已有知识图谱用于融合");
    return;
  }
  showToast("知识图谱融合任务已提交");
}

function submitDbGraphBuild() {
  if (!graphDatabaseForm.connectionName.trim() || !graphDatabaseForm.tableNames.trim()) {
    showToast("请补充数据库连接和数据表信息");
    return;
  }
  showToast("数据库导出构图任务已提交");
}

function selectGraphLibrary(libraryId) {
  selectedGraphLibraryId.value = libraryId;
  selectedGraphVersionId.value = graphLibraries.value.find((item) => item.id === libraryId)?.versions[0]?.id || "";
}

function selectGraphVersion(versionId) {
  selectedGraphVersionId.value = versionId;
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

function onPromptFileChange(event) {
  promptFile.value = event.target.files?.[0] || null;
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
      prompt: "",
      filter_thinking: true,
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.9,
    },
    finetuned: {
      fine_tuned_model_id: trainableFineTunedModels.value[0]?.id || "",
      model_id: readyBaseModels.value[0]?.id || "",
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
  if (!promptFile.value && !promptForm.content.trim()) {
    showToast("请填写提示词内容或上传提示词文件");
    return;
  }
  const form = new FormData();
  Object.entries(promptForm).forEach(([key, value]) => {
    form.append(key, value);
  });
  if (promptFile.value) {
    form.append("prompt_file", promptFile.value);
  }

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

async function deleteAgentWorkflow(agent) {
  if (!window.confirm(`删除智能体 ${agent.name}？`)) return;
  await fetchJSON(`/api/agents/${agent.id}`, { method: "DELETE" });
  if (managedAgentId.value === agent.id) {
    managedAgentId.value = "";
    managedAgentTestResult.value = null;
  }
  showToast("智能体已删除");
  await loadAgentWorkflows();
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
  if (isGraphPage(currentPage.value)) {
    loadGraphBuilderConfig().catch((error) => showToast(error.message));
  }
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
          <button class="menu-parent" :class="{ active: ['graphCreate', 'graphFusion', 'graphDatabase', 'graphVersions'].includes(currentPage) }" type="button">
            <Database />
            <span>多层次知识图谱构建</span>
            <ChevronDown />
          </button>
          <div class="sub-menu">
            <button :class="{ active: currentPage === 'graphCreate' }" type="button" @click="switchPage('graphCreate')">
              <FileUp />
              <span>新建知识库</span>
            </button>
            <button :class="{ active: currentPage === 'graphDatabase' }" type="button" @click="switchPage('graphDatabase')">
              <Database />
              <span>数据库导出构建</span>
            </button>
            <button :class="{ active: currentPage === 'graphFusion' }" type="button" @click="switchPage('graphFusion')">
              <Layers />
              <span>图谱融合构建</span>
            </button>
            <button :class="{ active: currentPage === 'graphVersions' }" type="button" @click="switchPage('graphVersions')">
              <Archive />
              <span>版本管理</span>
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
                  <input id="prompt-file" type="file" accept=".txt,.md,.json" @change="onPromptFileChange" />
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

        <section v-if="currentPage === 'graphCreate'" class="graph-page">
          <section class="guide-card">
            <div class="section-title">能力概览</div>
            <div class="guide-steps graph-guide-steps">
              <div>
                <div class="guide-icon"><FileUp /></div>
                <strong>多源构图</strong>
                <span>从文档、表格、图片统一抽取实体与关系</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><Layers /></div>
                <strong>图谱融合</strong>
                <span>对已有知识图谱进行对齐、归并与溯源</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><Database /></div>
                <strong>数据抽取</strong>
                <span>从业务数据库导出主题数据并映射为图谱</span>
              </div>
              <i></i>
              <div>
                <div class="guide-icon"><Archive /></div>
                <strong>版本治理</strong>
                <span>按版本追踪图谱变化并查看可视化结构</span>
              </div>
            </div>
          </section>

          <section class="dashboard-grid">
            <div class="metric-card blue-tint">
              <div class="metric-icon"><Database /></div>
              <span>知识库数量</span>
              <strong>{{ graphLibraryCount }}</strong>
            </div>
            <div class="metric-card green-tint">
              <div class="metric-icon"><Layers /></div>
              <span>累计实体数</span>
              <strong>{{ graphEntityCount }}</strong>
            </div>
            <div class="metric-card orange-tint">
              <div class="metric-icon"><Activity /></div>
              <span>累计关系数</span>
              <strong>{{ graphRelationCount }}</strong>
            </div>
          </section>

          <div class="graph-builder-grid">
            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>知识库信息</h2>
                <span class="tag processing">文档 / 表格 / 图片</span>
              </div>
              <form class="train-form" @submit.prevent="submitGraphCreate">
                <div class="form-grid two">
                  <label>
                    <span>知识库名称</span>
                    <input v-model="graphCreateForm.name" placeholder="请输入知识库名称" />
                  </label>
                  <label>
                    <span>业务领域</span>
                    <input v-model="graphCreateForm.businessDomain" placeholder="例如：设备运维" />
                  </label>
                  <label>
                    <span>图谱层次策略</span>
                    <select v-model="graphCreateForm.layerStrategy">
                      <option value="文档层 + 实体层 + 关系层">文档层 + 实体层 + 关系层</option>
                      <option value="文档层 + 实体层 + 关系层 + 事件层">文档层 + 实体层 + 关系层 + 事件层</option>
                      <option value="对象层 + 结构层 + 规则层">对象层 + 结构层 + 规则层</option>
                    </select>
                  </label>
                  <label>
                    <span>说明</span>
                    <input v-model="graphCreateForm.description" placeholder="说明本次构图目标和范围" />
                  </label>
                </div>

                <div class="graph-upload-grid">
                  <label class="upload-field graph-upload-field">
                    <input type="file" multiple accept=".pdf,.doc,.docx,.txt,.md,.xls,.xlsx,.csv" @change="onCreateSourceFilesChange" />
                    <FileUp />
                    <span>上传文档与表格</span>
                  </label>
                  <label class="upload-field graph-upload-field">
                    <input type="file" multiple accept=".png,.jpg,.jpeg,.bmp" @change="onCreateImageFilesChange" />
                    <FileUp />
                    <span>上传图片资料</span>
                  </label>
                </div>

                <div class="graph-file-summary">
                  <div>
                    <strong>已选结构化 / 非结构化文件</strong>
                    <small>{{ createSourceFiles.length ? createSourceFiles.map((item) => item.name).join("、") : "暂无文件" }}</small>
                  </div>
                  <div>
                    <strong>已选图片</strong>
                    <small>{{ createImageFiles.length ? createImageFiles.map((item) => item.name).join("、") : "暂无图片" }}</small>
                  </div>
                </div>

                <section class="graph-define-card">
                  <div class="panel-title graph-mini-title">
                    <h2>最近分析任务</h2>
                    <span class="tag processing">{{ graphAnalysisTasks.length }} 条</span>
                  </div>
                  <div class="graph-task-list">
                    <button
                      v-for="item in graphAnalysisTasks"
                      :key="item.id"
                      class="graph-task-item"
                      :class="{ active: graphAnalysisTaskId === item.id }"
                      type="button"
                      @click="loadGraphAnalysisTask(item.id)"
                    >
                      <div>
                        <strong>{{ item.knowledge_base_name || item.filename || "未命名分析任务" }}</strong>
                        <small>{{ item.filename }} · {{ item.analysis_source }}</small>
                      </div>
                      <div class="graph-schema-actions">
                        <span class="tag" :class="item.status === '已确认' ? 'success' : 'danger'">{{ item.status }}</span>
                        <span class="tag">{{ formatDate(item.updated_at) }}</span>
                      </div>
                    </button>
                    <div v-if="!graphAnalysisTasks.length" class="empty-state">上传 CSV / Excel 后会自动生成分析任务记录</div>
                  </div>
                </section>

                <section class="graph-define-card">
                  <div class="panel-title graph-mini-title">
                    <h2>原始表头展示</h2>
                    <span class="tag processing">{{ graphRawHeaders.length }} 个字段</span>
                  </div>
                  <div class="graph-header-grid">
                    <div v-for="item in graphRawHeaders" :key="item.id" class="graph-header-card">
                      <strong>{{ item.name }}</strong>
                      <small>{{ item.role }}</small>
                      <p>示例值：{{ item.sample }}</p>
                      <em>{{ item.description }}</em>
                    </div>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th v-for="item in graphRawHeaders" :key="item.id">{{ item.name }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, index) in graphSampleRows" :key="`sample-${index}`">
                          <td v-for="item in graphRawHeaders" :key="`${item.id}-${index}`">{{ row[item.name] }}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                <div class="graph-define-grid">
                  <section class="graph-define-card">
                    <div class="panel-title graph-mini-title">
                      <h2>模型建议的节点</h2>
                      <span class="tag processing">{{ graphModelNodeSuggestions.length }} 条</span>
                    </div>
                    <div class="graph-schema-list">
                      <div v-for="item in graphModelNodeSuggestions" :key="item.id" class="graph-schema-item">
                        <div>
                          <strong>{{ item.header }}</strong>
                          <small>建议节点类型：{{ item.node_type }} · {{ item.reason }}</small>
                        </div>
                        <span class="tag" :class="item.confidence === '高' ? 'success' : 'processing'">{{ item.confidence }}</span>
                      </div>
                    </div>
                  </section>

                  <section class="graph-define-card">
                    <div class="panel-title graph-mini-title">
                      <h2>模型建议的关系</h2>
                      <span class="tag processing">{{ graphModelRelationSuggestions.length }} 条</span>
                    </div>
                    <div class="graph-schema-list">
                      <div v-for="item in graphModelRelationSuggestions" :key="item.id" class="graph-schema-item">
                        <div>
                          <strong>{{ item.source_header }} -> {{ item.relation }} -> {{ item.target_header }}</strong>
                          <small>{{ item.reason }}</small>
                        </div>
                        <span class="tag" :class="item.confidence === '高' ? 'success' : 'processing'">{{ item.confidence }}</span>
                      </div>
                    </div>
                  </section>
                </div>

                <section class="graph-define-card">
                  <div class="panel-title graph-mini-title">
                    <h2>用户确认修改区</h2>
                    <div class="graph-schema-actions">
                      <span class="tag danger">{{ graphHeaderConfirmations.filter((item) => item.status === "待确认").length }} 条待确认</span>
                      <button class="secondary-button" type="button" @click="addGraphHeaderConfirmation">
                        <Plus />
                        <span>新增确认规则</span>
                      </button>
                    </div>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>起点表头</th>
                          <th>起点节点类型</th>
                          <th>关系</th>
                          <th>终点表头</th>
                          <th>终点节点类型</th>
                          <th>状态</th>
                          <th class="action-col">操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="item in graphHeaderConfirmations" :key="item.id">
                          <td>
                            <select v-model="item.source_header">
                              <option v-for="header in graphHeaderOptions" :key="`source-${item.id}-${header}`" :value="header">{{ header }}</option>
                            </select>
                          </td>
                          <td>
                            <select v-model="item.source_type">
                              <option v-for="nodeType in graphNodeTypes" :key="`source-type-${item.id}-${nodeType.id}`" :value="nodeType.name">
                                {{ nodeType.name }}
                              </option>
                            </select>
                          </td>
                          <td>
                            <select v-model="item.relation">
                              <option v-for="relationType in graphRelationTypes" :key="`relation-${item.id}-${relationType.id}`" :value="relationType.name">
                                {{ relationType.name }}
                              </option>
                            </select>
                          </td>
                          <td>
                            <select v-model="item.target_header">
                              <option v-for="header in graphHeaderOptions" :key="`target-${item.id}-${header}`" :value="header">{{ header }}</option>
                            </select>
                          </td>
                          <td>
                            <select v-model="item.target_type">
                              <option v-for="nodeType in graphNodeTypes" :key="`target-type-${item.id}-${nodeType.id}`" :value="nodeType.name">
                                {{ nodeType.name }}
                              </option>
                            </select>
                          </td>
                          <td><span class="tag" :class="item.status === '已确认' ? 'success' : 'danger'">{{ item.status }}</span></td>
                          <td class="action-col graph-triple-actions">
                            <button class="text-button" type="button" @click="confirmGraphHeaderRule(item.id)">确认</button>
                            <button class="text-button danger-text" type="button" @click="deleteGraphHeaderRule(item.id)">删除</button>
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                <section class="graph-define-card">
                  <div class="panel-title graph-mini-title">
                    <h2>三元组预览区</h2>
                    <span class="tag success">{{ graphTriplePreviewItems.length }} 条</span>
                  </div>
                  <div class="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>来源行</th>
                          <th>起点节点</th>
                          <th>关系</th>
                          <th>终点节点</th>
                          <th>状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="item in graphTriplePreviewItems" :key="item.id">
                          <td>{{ item.row_label }}</td>
                          <td>
                            <strong>{{ item.source }}</strong>
                            <small>{{ item.source_type }}</small>
                          </td>
                          <td><span class="tag processing">{{ item.relation }}</span></td>
                          <td>
                            <strong>{{ item.target }}</strong>
                            <small>{{ item.target_type }}</small>
                          </td>
                          <td><span class="tag" :class="item.status === '已确认' ? 'success' : 'danger'">{{ item.status }}</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                <div class="form-actions">
                  <button class="primary-button" type="submit">
                    <PackageCheck />
                    <span>开始构建</span>
                  </button>
                </div>
              </form>
            </section>

              <section class="panel-card graph-inner-card">
                <div class="panel-title">
                  <h2>构建知识图谱流程建议</h2>
                  <span class="tag">模型抽取 + 人工确认</span>
                </div>
                <div class="graph-process-list">
                  <div>
                    <strong>1. 上传原始业务资料</strong>
                    <small>优先上传真实业务表格、设备台账、故障记录、工艺记录等原始数据，不要求用户提前整理成三元组；文档、表格、图片分别接入，保留来源文件信息。</small>
                  </div>
                  <div>
                    <strong>2. 大模型识别表头语义</strong>
                    <small>由大模型理解字段含义，判断哪些表头适合作为设备、故障、组织、备件、位置等节点，哪些字段更适合作为属性或关系映射依据，并给出识别置信度。</small>
                  </div>
                  <div>
                    <strong>3. 自动生成候选关系</strong>
                    <small>基于表头语义和样例数据，自动推断字段之间可能存在的关系，例如“设备名称 -> 出现 -> 故障现象”“责任班组 -> 负责 -> 设备名称”，形成候选规则供审核。</small>
                  </div>
                  <div>
                    <strong>4. 用户确认并修正规则</strong>
                    <small>由业务人员确认起点表头、终点表头、节点类型和关系名称；对识别不准确的内容可直接修改、删除或新增，确保关系两端节点定义符合现场业务语义。</small>
                  </div>
                  <div>
                    <strong>5. 生成三元组预览并校验</strong>
                    <small>系统根据确认后的规则，从样例数据中生成三元组预览，展示每条关系的来源行、起点节点、关系、终点节点和状态，便于入库前核查准确性。</small>
                  </div>
                  <div>
                    <strong>6. 确认入库并保留任务记录</strong>
                    <small>确认后的节点、关系和三元组作为本次构图任务结果入库，同时保留抽取任务、确认规则、来源文件和预览结果，方便后续追溯、版本管理和图谱融合。</small>
                  </div>
                </div>
                <div class="graph-highlight-card">
                  <h3>推荐输出内容</h3>
                  <ul class="graph-inline-list">
                    <li>表头语义识别</li>
                    <li>候选节点清单</li>
                    <li>候选关系规则</li>
                    <li>三元组预览</li>
                    <li>任务追溯记录</li>
                  </ul>
                </div>
              </section>
          </div>
        </section>

        <section v-if="currentPage === 'graphFusion'" class="graph-page">
          <section class="graph-fusion-grid">
            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>选择已有知识库</h2>
                <span class="tag processing">{{ selectedFusionGraphIds.length }} 个已选</span>
              </div>
              <div class="graph-check-list">
                <button
                  v-for="item in graphLibraries"
                  :key="item.id"
                  class="graph-check-card"
                  :class="{ selected: activeFusionLibraryId === item.id }"
                  type="button"
                  @click="setActiveFusionLibrary(item.id)"
                >
                  <div>
                    <strong>{{ item.name }}</strong>
                    <small>{{ item.domain }} · {{ item.source }}</small>
                  </div>
                  <span class="tag">{{ item.children?.length || 0 }} 个子库</span>
                </button>
              </div>
              <div class="graph-sub-library-panel">
                <div class="graph-sub-library-header">
                  <div>
                    <strong>{{ activeFusionLibrary?.name || "-" }}</strong>
                    <small>请选择下方子知识库参与融合</small>
                  </div>
                  <span class="tag processing">{{ activeFusionChildren.length }} 个子库</span>
                </div>
                <div class="search-field graph-search-field">
                  <Search />
                  <input v-model="fusionSearch" placeholder="搜索子知识库名称或场景" />
                </div>
                <div class="graph-sub-library-list">
                  <button
                    v-for="child in activeFusionChildren"
                    :key="child.id"
                    class="graph-sub-library-card"
                    :class="{ selected: selectedFusionGraphIds.includes(child.id) }"
                    type="button"
                    @click="toggleFusionGraph(child.id)"
                  >
                    <div>
                      <strong>{{ child.name }}</strong>
                      <small>{{ child.scene }}</small>
                    </div>
                    <div class="graph-sub-library-meta">
                      <span class="tag">{{ child.entity_count }} 实体</span>
                      <span class="tag processing">{{ child.relation_count }} 关系</span>
                    </div>
                  </button>
                </div>
                <div class="graph-selected-summary">
                  <div class="panel-title graph-mini-title">
                    <h2>已选子知识库汇总</h2>
                    <span class="tag success">{{ fusionSourceGraphs.length }} 个</span>
                  </div>
                  <div v-if="fusionSourceGraphs.length" class="graph-selected-list">
                    <div v-for="item in fusionSourceGraphs" :key="item.id" class="graph-selected-item">
                      <div>
                        <strong>{{ item.name }}</strong>
                        <small>{{ item.parent_name }} · {{ item.scene }}</small>
                      </div>
                      <div class="graph-sub-library-meta">
                        <span class="tag">{{ item.entity_count }} 实体</span>
                        <span class="tag processing">{{ item.relation_count }} 关系</span>
                      </div>
                    </div>
                  </div>
                  <div v-else class="empty-state">请选择至少两个子知识库</div>
                </div>
              </div>
            </section>

            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>融合策略</h2>
                <span class="tag">实体对齐</span>
              </div>
              <form class="train-form" @submit.prevent="submitGraphFusion">
                <div class="form-grid two">
                  <label>
                    <span>新图谱名称</span>
                    <input v-model="graphFusionForm.name" placeholder="请输入融合后的图谱名称" />
                  </label>
                  <label>
                    <span>融合模式</span>
                    <select v-model="graphFusionForm.fusionMode">
                      <option value="实体对齐优先">实体对齐优先</option>
                      <option value="关系完整优先">关系完整优先</option>
                      <option value="版本溯源优先">版本溯源优先</option>
                    </select>
                  </label>
                  <label>
                    <span>冲突处理规则</span>
                    <input v-model="graphFusionForm.conflictRule" placeholder="例如：保留高可信版本" />
                  </label>
                  <label>
                    <span>说明</span>
                    <input v-model="graphFusionForm.description" placeholder="描述本次融合目标" />
                  </label>
                </div>

                <div class="graph-fusion-preview">
                  <strong>待融合子知识库</strong>
                  <div class="graph-chip-row">
                    <span v-for="item in fusionSourceGraphs" :key="item.id" class="tag processing">{{ item.parent_name }} / {{ item.name }}</span>
                  </div>
                </div>

                <div class="graph-fusion-preview">
                  <strong>融合前检查</strong>
                  <div class="graph-chip-row">
                    <span class="tag processing">实体对齐候选 {{ graphAlignmentPreview.length }}</span>
                    <span class="tag">{{ graphAlignmentPendingCount }} 条待确认</span>
                    <span class="tag danger">高优先冲突 {{ graphConflictHighCount }}</span>
                  </div>
                </div>

                <div class="form-actions">
                  <button class="primary-button" type="submit">
                    <Layers />
                    <span>提交融合构建</span>
                  </button>
                </div>
              </form>

              <div class="graph-detail-grid">
                <section class="graph-detail-card">
                  <div class="panel-title graph-mini-title">
                    <h2>实体对齐预览</h2>
                    <span class="tag processing">{{ graphAlignmentPreview.length }} 条</span>
                  </div>
                  <div class="graph-detail-list">
                    <div v-for="item in graphAlignmentPreview" :key="item.id" class="graph-detail-item">
                      <div class="graph-detail-main">
                        <strong>{{ item.left }}</strong>
                        <small>{{ item.right }}</small>
                      </div>
                      <div class="graph-detail-meta">
                        <span class="tag">{{ item.confidence }}</span>
                        <span class="tag processing">{{ item.rule }}</span>
                        <span class="tag" :class="item.decision === '待确认' ? 'danger' : 'success'">{{ item.decision }}</span>
                      </div>
                    </div>
                  </div>
                </section>

                <section class="graph-detail-card">
                  <div class="panel-title graph-mini-title">
                    <h2>冲突项待确认</h2>
                    <span class="tag danger">{{ graphConflictItems.length }} 项</span>
                  </div>
                  <div class="graph-detail-list">
                    <div v-for="item in graphConflictItems" :key="item.id" class="graph-detail-item">
                      <div class="graph-detail-main">
                        <strong>{{ item.entity }} · {{ item.field }}</strong>
                        <small>{{ item.left }} / {{ item.right }}</small>
                      </div>
                      <div class="graph-detail-meta">
                        <span class="tag" :class="item.severity === '高' ? 'danger' : 'processing'">{{ item.severity }}优先级</span>
                        <span class="tag">{{ item.suggestion }}</span>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            </section>
          </section>
        </section>

        <section v-if="currentPage === 'graphDatabase'" class="graph-page">
          <div class="graph-builder-grid">
            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>数据库构图配置</h2>
                <span class="tag processing">主数据抽取</span>
              </div>
              <form class="train-form" @submit.prevent="submitDbGraphBuild">
                <div class="form-grid two">
                  <label>
                    <span>知识库名称</span>
                    <input v-model="graphDatabaseForm.name" placeholder="请输入知识库名称" />
                  </label>
                  <label>
                    <span>数据库类型</span>
                    <select v-model="graphDatabaseForm.databaseType">
                      <option value="PostgreSQL">PostgreSQL</option>
                      <option value="MySQL">MySQL</option>
                      <option value="SQL Server">SQL Server</option>
                      <option value="Oracle">Oracle</option>
                    </select>
                  </label>
                  <label>
                    <span>连接名称</span>
                    <input v-model="graphDatabaseForm.connectionName" placeholder="请输入只读连接名称" />
                  </label>
                  <label>
                    <span>Schema / 库</span>
                    <input v-model="graphDatabaseForm.schemaName" placeholder="public" />
                  </label>
                  <label>
                    <span>数据表</span>
                    <input v-model="graphDatabaseForm.tableNames" placeholder="多个表以逗号分隔" />
                  </label>
                  <label>
                    <span>同步模式</span>
                    <select v-model="graphDatabaseForm.syncMode">
                      <option value="按主题导出后构图">按主题导出后构图</option>
                      <option value="全量导出后构图">全量导出后构图</option>
                      <option value="增量抽取构图">增量抽取构图</option>
                    </select>
                  </label>
                </div>

                <div class="form-actions">
                  <button class="primary-button" type="submit">
                    <Database />
                    <span>开始导出构图</span>
                  </button>
                </div>
              </form>
            </section>

            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>抽取范围</h2>
                <span class="tag">数据库表映射</span>
              </div>
              <div class="graph-process-list">
                <div>
                  <strong>设备主数据</strong>
                  <small>设备编码、位置、责任组织、设备类型等基础主数据。</small>
                </div>
                <div>
                  <strong>运维事件</strong>
                  <small>工单、保养策略、故障标签、执行记录等事件链路。</small>
                </div>
                <div>
                  <strong>备件与物料</strong>
                  <small>备件编码、替代件、供应关系、库存状态。</small>
                </div>
              </div>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'graphVersions'" class="graph-page">
          <div class="graph-version-layout">
            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>知识库列表</h2>
                <button class="secondary-button" type="button" @click="switchPage('graphCreate')">
                  <Plus />
                  <span>新增构图</span>
                </button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>知识库</th>
                      <th>来源</th>
                      <th>层次</th>
                      <th>更新时间</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="item in latestGraphLibraries"
                      :key="item.id"
                      :class="{ selected: selectedGraphLibraryId === item.id }"
                      @click="selectGraphLibrary(item.id)"
                    >
                      <td>
                        <strong>{{ item.name }}</strong>
                        <small>{{ item.domain }} · {{ item.owner }}</small>
                      </td>
                      <td>{{ item.source }}</td>
                      <td><span class="tag processing">{{ item.layers.length }} 层</span></td>
                      <td>{{ formatDate(item.updated_at) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section class="panel-card graph-inner-card">
              <div class="panel-title">
                <h2>版本与可视化</h2>
                <span class="tag success">{{ currentGraphVersion?.label || "未选择" }}</span>
              </div>
              <div class="graph-version-content">
                <div class="graph-version-sidebar">
                  <button
                    v-for="version in currentGraphLibrary?.versions || []"
                    :key="version.id"
                    class="graph-version-button"
                    :class="{ active: selectedGraphVersionId === version.id }"
                    type="button"
                    @click="selectGraphVersion(version.id)"
                  >
                    <strong>{{ version.label }}</strong>
                    <small>{{ formatDate(version.updated_at) }}</small>
                  </button>
                </div>
                <div class="graph-visual-panel">
                  <div class="graph-visual-summary">
                    <div>
                      <strong>{{ currentGraphLibrary?.name || "-" }}</strong>
                      <small>{{ currentGraphVersion?.summary || "请选择版本查看" }}</small>
                    </div>
                    <div class="graph-chip-row">
                      <span class="tag processing">实体 {{ currentGraphVersion?.metrics?.entities || 0 }}</span>
                      <span class="tag">关系 {{ currentGraphVersion?.metrics?.relations || 0 }}</span>
                      <span class="tag">来源 {{ currentGraphVersion?.metrics?.sources || 0 }}</span>
                    </div>
                  </div>

                  <div class="graph-canvas-mini">
                    <div v-for="node in currentGraphVersion?.nodes || []" :key="node.id" class="graph-node-card">
                      <strong>{{ node.label }}</strong>
                      <small>{{ node.type }}</small>
                    </div>
                  </div>

                  <div class="graph-edge-list">
                    <strong>关系链路</strong>
                    <div v-for="edge in currentGraphVersion?.edges || []" :key="edge">
                      <small>{{ edge }}</small>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </section>

        <section v-if="currentPage === 'agentManage'" class="agent-manage-page">
          <div class="agent-manage-grid">
            <section class="panel-card">
              <div class="panel-title">
                <h2>智能体列表</h2>
                <button class="secondary-button" type="button" @click="loadAgentWorkflows"><RefreshCw />刷新</button>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>智能体名称</th>
                      <th>节点数</th>
                      <th>状态</th>
                      <th>更新时间</th>
                      <th class="action-col">操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-if="!agentWorkflows.length">
                      <td colspan="5" class="empty">暂无智能体</td>
                    </tr>
                    <tr v-for="agent in agentWorkflows" :key="agent.id" :class="{ selected: managedAgentId === agent.id }">
                      <td>
                        <strong>{{ agent.name }}</strong>
                        <small>{{ agent.description || "顺序执行链路" }}</small>
                      </td>
                      <td>{{ agent.node_count || agent.nodes?.length || 0 }}</td>
                      <td><span class="tag success">{{ statusText(agent.status) }}</span></td>
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
