# realtime_dltraining_system

This project is a traffic data imputation system based on deep learning. It applies the SAITS (Self-Attention-based Imputation for Time Series) algorithm to intelligently complete missing traffic flow data and perform visual analysis.
In the training interface, users can observe **real-time changes** in **accuracy and loss** as the model trains. After training, the platform **automatically generates an interactive visualization** that presents the algorithm’s prediction results in a clear and intuitive way.

## Project structure

### backend/main.py

The main file of the FastAPI application, including WebSocket routes and static file service

### backend/saits_service.py

SAITS algorithm encapsulation, handling training logic and progress push

### backend/static

Save the original dataset **pems.npy**, but it is private data, so it do not contain in here. You can put your data in this folder.

### src/components/Training.vue

The main training interface includes the configuration panel, training monitoring, and result visualization

### public/

save some public files in here.

## Environment & Operating way

Python 3.8+ & Node.js 14+

backend tools:  FastAPI, PyPOTS, PyTorch, scikit-learn

frontend tools: Vue3, Plotly.js, WebSocket



Run in the command prompt environment under the "backend" folder:

```cmd
uvicorn main:app --reload
```

Run in the command prompt environment under the root directory of the project file:

```cmd
npm run dev
```

## 

## Usage example

This is the initial interface of the project, featuring three functional windows: training settings, training controller, and result display.

<img src="public\1.png" alt="1" style="zoom:23%;" />

### Training Configuration

On this interface, you can freely select the dataset and deep learning algorithm model for training.

<img src="public\2.png" alt="1" style="zoom:30%;" />

### Model Training Monitor

On this interface, you can click "Start Training" and observe the real-time training accuracy and loss.

<img src="public\3.png" alt="1" style="zoom:24.5%;" />

### Results showing

In this interface, you can compare the differences between the original time series data and the predicted data.

<img src="public\4.png" alt="1" style="zoom:26.5%;" />

