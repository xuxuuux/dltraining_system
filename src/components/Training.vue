<template>
  <div class="outer-wrapper">
    <div class="train-container">
      <!-- 主标题区域 -->
      <div class="main-header">
        <h1 class="main-title">Traffic DeepL Training System</h1>
        <p class="main-subtitle">Advanced artificial intelligence algorithm training and visualization</p>
        <div class="header-divider"></div>
      </div>

      <!-- 算法和数据集选择面板 -->
      <div class="collapsible">
        <div class="collapsible-header" @click="openConfig = !openConfig">
          <h2>Training Configuration</h2>
          <span class="arrow" :class="{ open: openConfig }">▼</span>
        </div>

        <div v-show="openConfig" class="collapsible-content">
          <div class="config-card">
            <header class="header">
              <h1>Training Configuration</h1>
              <p class="subtitle">Select Algorithm and Dataset for Training</p>
            </header>

            <div class="config-controls">
              <div class="config-group">
                <label for="algorithmSelect">Algorithm:</label>
                <select id="algorithmSelect" v-model="selectedAlgorithm" disabled>
                  <option value="saits">SAITS (Self-Attention-based Imputation)</option>
                </select>
                <span class="info-text">Currently only SAITS algorithm is available</span>
              </div>

              <div class="config-group">
                <label for="datasetSelect">Dataset:</label>
                <select id="datasetSelect" v-model="selectedDataset" disabled>
                  <option value="pems">PEMS Traffic Flow Dataset</option>
                </select>
                <span class="info-text">Currently only PEMS dataset is available</span>
              </div>

              <div class="config-group">
                <label for="missingRate">Missing Rate:</label>
                <input 
                  id="missingRate" 
                  type="range" 
                  min="5" 
                  max="50" 
                  step="5" 
                  v-model.number="missingRate" 
                  class="slider"
                  disabled
                >
                <span class="slider-value">{{ missingRate }}%</span>
                <span class="info-text">Missing rate configuration will be available in the future</span>
              </div>
            </div>

            <div class="config-summary">
              <h3>Current Configuration</h3>
              <div class="summary-items">
                <div class="summary-item">
                  <span class="label">Algorithm:</span>
                  <span class="value">SAITS</span>
                </div>
                <div class="summary-item">
                  <span class="label">Dataset:</span>
                  <span class="value">{{ selectedDataset }}</span>
                </div>
                <div class="summary-item">
                  <span class="label">Missing Rate:</span>
                  <span class="value">{{ missingRate }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Collapsible Panel: Model Training Monitor -->
      <div class="collapsible">
        <div class="collapsible-header" @click="openTraining = !openTraining">
          <h2>Model Training Monitor</h2>
          <span class="arrow" :class="{ open: openTraining }">▼</span>
        </div>

        <div v-show="openTraining" class="collapsible-content">
          <div class="training-card">
            <header class="header">
              <h1>Model Training Monitor</h1>
              <p class="subtitle">Real-time Tracking of Epoch, Loss and Accuracy Changes</p>
            </header>

            <div class="controls">
              <button @click="startTraining" :disabled="wsOpen" class="btn start">
                <span>Start Training</span>
              </button>
              <button @click="stopTraining" :disabled="!wsOpen" class="btn stop">
                <span>Stop Training</span>
              </button>

              <div class="metrics">
                <div class="metric">
                  <span class="label">Epoch</span>
                  <span class="value">{{ epoch }}</span>
                </div>
                <div class="metric">
                  <span class="label">Loss</span>
                  <span class="value loss">{{ loss.toFixed(4) }}</span>
                </div>
                <div class="metric">
                  <span class="label">Accuracy</span>
                  <span class="value acc">{{ accuracy.toFixed(4) }}</span>
                </div>
              </div>
            </div>

            <div class="progress-bar">
              <div class="progress" :style="{ width: `${Math.min((epoch/100)*100, 100)}%` }"></div>
            </div>

            <div id="training-chart" class="chart-box"></div>
          </div>
        </div>
      </div>

      <!-- Collapsible Panel: Imputation Results Visualization -->
      <div class="collapsible">
        <div class="collapsible-header" @click="openImputation = !openImputation">
          <h2>Imputation Results Visualization</h2>
          <span class="arrow" :class="{ open: openImputation }">▼</span>
        </div>

        <div v-show="openImputation" class="collapsible-content">
          <div class="imputation-container">
            <header class="header">
              <h1>Imputation Results Visualization</h1>
              <p class="subtitle">Compare Original, Missing and Imputed Traffic Flow Data</p>
            </header>

            <!-- 添加条件渲染，只有训练完成后才显示内容 -->
            <div v-if="imputationDataLoaded" class="imputation-content">
              <section class="control-panel">
                <label for="featureSelect">Select Feature Node:</label>
                <select id="featureSelect" v-model.number="selectedFeature" @change="updateImputationPlot">
                  <option v-for="i in numFeatures" :key="i" :value="i - 1">
                    Feature {{ i - 1 }}
                  </option>
                </select>
              </section>

              <section class="chart-section">
                <div ref="chart" class="chart"></div>
              </section>
            </div>
            
            <!-- 训练完成前显示占位内容 -->
            <div v-else class="placeholder-content">
              <p class="placeholder-text">Training in progress, and the result will be displayed here after training is complete.</p>
              <div class="loading-animation">
                <div class="spinner"></div>
                <span>Waiting for training data...</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 页脚 -->
      <footer class="app-footer">
        <!-- 页脚内容留空 -->
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import Plotly from 'plotly.js-dist'
import NPY from 'npyjs'

/* Collapse Control */
const openConfig = ref(true)
const openTraining = ref(true)
const openImputation = ref(false)

/* Configuration Variables */
const selectedAlgorithm = ref('saits')
const selectedDataset = ref('pems')
const missingRate = ref(10)

/* Training State Variables */
const epoch = ref(0)
const loss = ref(0)
const accuracy = ref(0)
const wsOpen = ref(false)
const imputationDataLoaded = ref(false)

const selectedFeature = ref(0)
const numFeatures = ref(1)

let dataArr = null
let imputedArr = null
let incompleteArr = null
let timeSteps = 0
let features = 0

const WS_URL = 'ws://127.0.0.1:8000/ws/train'
const MISSING_RATE = 0.1
const RNG_SEED = 42
let ws = null

function initTrainingPlot() {
  const div = document.getElementById('training-chart')
  Plotly.newPlot(div, [
    { x: [], y: [], type: 'scatter', mode: 'lines+markers', name: 'Loss', line: { color: '#8b0000', width: 2 } },
    { x: [], y: [], type: 'scatter', mode: 'lines+markers', name: 'Accuracy', yaxis: 'y2', line: { color: '#2e8b57', width: 2 } }
  ], {
    title: 'Training Process (Real-time)',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss' },
    yaxis2: { title: 'Accuracy', overlaying: 'y', side: 'right', range: [0, 1] },
    legend: { orientation: 'h', bgcolor: '#fff', bordercolor: '#ddd' },
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#fafafa'
  })
}

function extendTrainingPlot(ep, lossVal, accVal) {
  const div = document.getElementById('training-chart')
  Plotly.extendTraces(div, { x: [[ep], [ep]], y: [[lossVal], [accVal]] }, [0, 1])
}

function seededRandom(seed) {
  let x = Math.sin(seed++) * 10000
  return function() {
    x = Math.sin(x) * 10000
    return x - Math.floor(x)
  }
}

function generateIncomplete() {
  incompleteArr = { ...dataArr, data: new Float32Array(dataArr.data) }
  const rng = seededRandom(RNG_SEED)
  for (let t = 0; t < timeSteps; t++) {
    for (let f = 0; f < features; f++) {
      if (rng() < MISSING_RATE) incompleteArr.data[t * features + f] = NaN
    }
  }
}

async function updateImputationPlot() {
  if (!dataArr || !imputedArr || !incompleteArr) return

  await nextTick()
  const f = selectedFeature.value
  const original = []
  const incomplete = []
  const imputed = []

  for (let t = 0; t < timeSteps; t++) {
    original.push(dataArr.data[t * features + f])
    incomplete.push(incompleteArr.data[t * features + f])
    imputed.push(imputedArr.data[t * features + f])
  }

  const traces = [
    { y: original, type: 'scatter', mode: 'lines', name: 'Original', line: { color: '#4682b4', width: 2 } },
    { y: incomplete, type: 'scatter', mode: 'lines+markers', name: 'With Missing', line: { color: '#8b0000', width: 2 }, marker: { size: 4 } },
    { y: imputed, type: 'scatter', mode: 'lines', name: 'Imputed', line: { color: '#2e8b57', width: 2 } }
  ]

  const layout = {
    title: `Traffic Data - Feature ${f}`,
    xaxis: { title: 'Time Step', rangeslider: { visible: true, thickness: 0.05 } },
    yaxis: { title: 'Value' },
    template: 'plotly_white',
    height: 600,
    margin: { l: 80, r: 50, t: 80, b: 80 },
    font: { family: 'Times New Roman, serif', size: 14 },
    legend: { orientation: 'h', y: -0.2 }
  }

  Plotly.newPlot(document.querySelector('.chart'), traces, layout, { responsive: true })
}

function stopTraining() {
  if (ws) {
    try { ws.close() } catch {}
    ws = null
    wsOpen.value = false
  }
}

async function startTraining() {
  initTrainingPlot()
  epoch.value = 0
  loss.value = 0
  accuracy.value = 0
  imputationDataLoaded.value = false

  ws = new WebSocket(WS_URL)
  ws.onopen = () => { wsOpen.value = true }

  ws.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data)

    if (msg.epoch !== undefined) {
      epoch.value = msg.epoch
      loss.value = msg.loss
      accuracy.value = msg.accuracy ?? Math.max(0, 1 - msg.loss)
      extendTrainingPlot(msg.epoch, msg.loss, accuracy.value)
    }

    if (msg.status === 'done') {
      wsOpen.value = false
      try { ws.close() } catch {}
      ws = null

      const npy = new NPY()
      const [dataResp, imputedResp] = await Promise.all([
        fetch('/pems.npy'),
        fetch('/imputed.npy')
      ])
      dataArr = await npy.load(await dataResp.arrayBuffer())
      imputedArr = await npy.load(await imputedResp.arrayBuffer())

      const shape = dataArr.shape
      if (shape.length === 3) {
        timeSteps = shape[1]
        features = shape[2]
      } else if (shape.length === 2) {
        timeSteps = shape[0]
        features = shape[1]
      }
      numFeatures.value = features

      generateIncomplete()
      imputationDataLoaded.value = true
      selectedFeature.value = 0
      openImputation.value = true
      await updateImputationPlot()
    }

    if (msg.status === 'error') alert('Server Error: ' + msg.message)
  }

  ws.onclose = () => { wsOpen.value = false }
  ws.onerror = (err) => { wsOpen.value = false; console.error('WS error', err) }
}

window.addEventListener('beforeunload', () => { if (ws) ws.close() })
</script>

<style>
/* 基础样式 - 专业心理学配色方案 */
.outer-wrapper {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  min-height: 100vh;
  background: #f8f9fa;
  padding: 20px;
  position: relative;
  overflow: hidden;
}

.train-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 95%;
  max-width: 1400px;
  padding: 20px;
  font-family: "Times New Roman", serif;
  position: relative;
  z-index: 1;
}

/* 主标题区域 */
.main-header {
  text-align: center;
  margin-bottom: 40px;
  width: 100%;
}

.main-title {
  font-size: 2.2rem;
  font-weight: 700;
  color: #2f4f4f; /* 深灰色 - 专业稳重 */
  margin-bottom: 10px;
  letter-spacing: 0.5px;
}

.main-subtitle {
  font-size: 1.1rem;
  color: #696969; /* 中灰色 - 专业但不压抑 */
  margin-bottom: 20px;
  font-style: italic;
}

.header-divider {
  height: 2px;
  width: 80px;
  background: #008b8b; /* 深青色 - 专业可靠 */
  margin: 0 auto;
}

/* Collapsible Cards */
.collapsible {
  width: 100%;
  background: white;
  border-radius: 8px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  overflow: hidden;
  transition: all 0.3s ease;
  border: 1px solid #dcdcdc;
}

.collapsible-header {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: #2f4f4f; /* 深灰色 - 专业稳重 */
  color: white;
  font-size: 1rem;
  font-weight: 600;
  user-select: none;
  height: 60px;
  box-sizing: border-box;
  transition: all 0.3s ease;
}

.collapsible-header:hover {
  background: #3a5f5f; /* 稍亮的深灰色 */
}

.collapsible-header h2 {
  font-size: 1.1rem;
  margin: 0;
  flex: 1;
}

.arrow {
  transition: transform 0.3s ease;
  font-size: 0.9rem;
}
.arrow.open {
  transform: rotate(180deg);
}

.collapsible-content {
  background: transparent;
  padding: 20px;
  box-sizing: border-box;
  width: 100%;
}

/* 标题样式 */
.training-card, .imputation-container, .config-card {
  width: 100%;
  border-radius: 6px;
  background: #ffffff;
  padding: 20px;
  box-sizing: border-box;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  border: 1px solid #e8e8e8;
}

.header {
  text-align: center;
  margin-bottom: 20px;
}

.header h1 {
  color: #2f4f4f; /* 深灰色 */
  font-weight: 600;
  font-size: 1.4rem;
  margin-bottom: 8px;
}

.subtitle {
  color: #696969; /* 中灰色 */
  margin-top: 8px;
  font-size: 1rem;
}

/* 配置控制面板样式 */
.config-controls {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-bottom: 25px;
}

.config-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e8e8e8;
}

.config-group label {
  font-weight: 600;
  color: #2f4f4f; /* 深灰色 */
  font-size: 1rem;
}

.config-group select {
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #c0c0c0;
  font-size: 0.95rem;
  background: white;
  transition: all 0.2s ease;
}

.config-group select:disabled {
  background-color: #f0f0f0;
  color: #808080;
  cursor: not-allowed;
}

.info-text {
  font-size: 0.8rem;
  color: #808080;
  font-style: italic;
}

/* 滑块样式 */
.slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  transition: opacity 0.2s;
}

.slider:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.slider-value {
  font-weight: 600;
  color: #2f4f4f; /* 深灰色 */
  min-width: 40px;
}

/* 配置摘要样式 */
.config-summary {
  background: #f0f8f8; /* 淡青色背景 */
  padding: 16px;
  border-radius: 6px;
  border: 1px solid #b0c4c4;
}

.config-summary h3 {
  margin-top: 0;
  margin-bottom: 12px;
  color: #2f4f4f;
  font-size: 1.1rem;
}

.summary-items {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  background: white;
  padding: 10px 14px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  min-width: 120px;
}

.summary-item .label {
  font-size: 0.8rem;
  color: #696969;
  margin-bottom: 4px;
}

.summary-item .value {
  font-size: 1rem;
  font-weight: 600;
  color: #2f4f4f;
}

/* 训练控制面板样式 */
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  background: #f8f9fa;
  padding: 16px;
  border-radius: 6px;
  border: 1px solid #e8e8e8;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  color: white;
  font-weight: 600;
  min-width: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.btn:active {
  transform: translateY(0);
}

.btn.start { 
  background: #008b8b; /* 深青色 - 专业可靠 */
}

.btn.start:hover { 
  background: #006f6f; 
}

.btn.stop { 
  background: #8b4513; /*  saddlebrown - 专业的中性停止色 */
}

.btn.stop:hover { 
  background: #734c12; 
}

.btn:disabled { 
  opacity: 0.6; 
  cursor: not-allowed; 
  transform: none;
  box-shadow: none;
}

.metrics {
  display: flex;
  gap: 20px;
  justify-content: center;
  font-size: 0.95rem;
  background: white;
  padding: 12px;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.metric {
  text-align: center;
  padding: 8px 12px;
  border-radius: 4px;
  background: #f8f9fa;
  min-width: 80px;
}

.metric .label { 
  display: block; 
  font-size: 0.8rem; 
  color: #696969; 
  margin-bottom: 4px;
  font-weight: 600;
}

.metric .value { 
  font-size: 1.1rem; 
  font-weight: bold; 
}

.metric .loss { color: #8b0000; } /* 深红色 */
.metric .acc { color: #2e8b57; } /* 海绿色 */

.progress-bar {
  height: 6px;
  background: #e8e8e8;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 16px;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
}
.progress {
  height: 6px;
  background: #4682b4; /* 钢蓝色 - 专业可靠 */
  transition: width 0.5s ease;
  border-radius: 3px;
}

.chart-box { 
  height: 300px; 
  width: 100%;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #e8e8e8;
}

.control-panel {
  margin-bottom: 20px;
  background: #f8f9fa;
  padding: 12px 16px;
  border-radius: 6px;
  display: flex;
  align-items: center;
}

.control-panel label {
  font-weight: 600;
  color: #2f4f4f;
}

.control-panel select {
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid #c0c0c0;
  font-size: 0.95rem;
  background: white;
  transition: all 0.2s ease;
  margin-left: 10px;
}
.control-panel select:hover { 
  border-color: #008b8b; 
}
.control-panel select:focus {
  outline: none;
  border-color: #008b8b;
  box-shadow: 0 0 0 2px rgba(0, 139, 139, 0.2);
}

.chart {
  width: 100%;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #e8e8e8;
}

/* 占位内容样式 */
.placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
  background: #f8f9fa;
  border-radius: 6px;
  border: 1px dashed #c0c0c0;
}

.placeholder-text {
  color: #696969;
  font-size: 1rem;
  margin-bottom: 20px;
  max-width: 400px;
  line-height: 1.4;
}

.loading-animation {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e8e8e8;
  border-top: 3px solid #008b8b; /* 深青色 */
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-animation span {
  color: #696969;
  font-size: 0.9rem;
}

/* 页脚样式 */
.app-footer {
  margin-top: 30px;
  padding: 20px;
  text-align: center;
  color: #696969;
  font-size: 0.85rem;
  width: 100%;
  border-top: 1px solid #e8e8e8;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .main-title {
    font-size: 1.8rem;
  }
  
  .controls {
    flex-direction: column;
    gap: 12px;
  }
  
  .metrics {
    flex-direction: column;
    gap: 12px;
  }
  
  .btn {
    width: 100%;
  }
  
  .summary-items {
    flex-direction: column;
  }
}
</style>