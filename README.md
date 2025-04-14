# Image Classification - Cats vs Dogs 🐱🐶

This project trains and evaluates deep learning models for **binary image classification** (Cat vs Dog), using **PyTorch**, **Optuna** for hyperparameter tuning, and visualizations with `matplotlib` and `great_tables`.

---

## 📁 Project Structure

```
.
├── notebook.ipynb              # Main notebook with training, evaluation, and visualization
├── demo_models.ipynb           # Just shows some predictions of some images
├── dataset.py                  # Defines dataset for DataLoader, allowing it to be parallelized on MacOS
├── models/                     # Trained models (.pt)
├── graphs/                     # Saved plots
├── requirements.txt            # Dependencies
```

---

## 🛠️ Requirements

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

> To save tables with `great_tables`, the `weasyprint` package is also required (and may need system dependencies like Cairo and Pango).

All of the data is imported using `kagglehub`.

---

## 🚀 How to Run

1. Make sure the images are in the `dataset/` folder, organized by class (e.g., `Cat`, `Dog`)
2. Open the Jupyter Notebook:

```bash
jupyter notebook notebook.ipynb
```

3. Run all cells in order or click `Kernel > Restart and Run All`.

---

## 🔄 How to Reproduce Results

- **Training**: the notebook trains custom CNN models (including fine-tuned ResNet-18) and saves the most accurate model to `models` folder.
- **Evaluation**: metrics like accuracy, loss, and ROC curves are automatically generated
- **Classification Report**: class-wise metric tables are rendered with `great_tables`.

---

## 📊 Results

- **Final accuracy on holdout**: 95%
- **Best model**: `ResNet-18`
- **Visualizations**:
  - Loss/accuracy per epoch
  - Feature maps from early layers
  - Confusion matrix and ROC curve
  - Classification report tables

---

## 📌 Notes

- Experiments use `Optuna` to search for the best **hyperparameters** and **architecture**.
- Data is loaded with `torchvision.transforms` and `DataLoader` with seed control for reproducibility