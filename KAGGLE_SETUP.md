# Kaggle Setup Guide for Business Entity Resolution

Follow these instructions to successfully train your ML pipeline on Kaggle, utilizing their free GPUs to speed up training and boost accuracy to 0.99+.

## 1. Upload Your Local Data to Kaggle

1. Go to Kaggle (https://www.kaggle.com) and sign in.
2. On the left sidebar, click **Datasets**, then click the **New Dataset** button (top right).
3. Name your dataset (for example: `amazon-ml-2026-data`).
4. Click **Browse Files** and upload ONLY the `6ab10eb3b23ba_student_resource.zip` file from your `d:\amazonml\data source given` folder. 
   *(Kaggle will automatically extract the zip file contents for you into the dataset!)*
5. Keep the visibility set to **Private**.
6. Click **Create** and wait for the upload to complete.

---

## 2. Push Your Local Code to GitHub

Before creating the Kaggle notebook, make sure all the code updates we made locally are pushed to your GitHub repository!

Run these commands in your local VS Code terminal:
```bash
git add .
git commit -m "Optimize XGBoost and LightGBM for GPU and better accuracy"
git push
```

---

## 3. Create & Configure the Kaggle Notebook

1. Go to Kaggle and click **+ Create** -> **New Notebook**.
2. **Turn on the GPU:** On the right-hand panel, click on **Notebook Options** or **Settings**. Change the **Accelerator** to **GPU T4 x2** (or GPU P100).
3. **Turn on Internet:** Ensure the **Internet** toggle is set to **ON** (required for cloning your repo).
4. **Attach your Data:** On the right-hand panel under **Input**, click **+ Add Input** (or Add Data). Switch to the **Your Datasets** tab, and click the `+` button next to the dataset you created in Step 1.

---

## 4. Run the Code

Copy and paste the following cells into your new Kaggle Notebook in this exact order:

### Cell 1: Clone Repository & Setup
```python
# Clone your updated GitHub repository
!git clone https://github.com/svmayurkaarthic2005/business-entity-resolution.git

# Move into the working directory
%cd business-entity-resolution

# Install required dependencies
!pip install -r requirements.txt
```

### Cell 2: Link the Input Data and Fix Config
*Note: Replace `amazon-ml-2026-data` with the exact name you used for your dataset in Step 1.*
```python
# Create the local data directory
!mkdir -p data

# Flatten and copy all .tsv files from the extracted zip into our single data folder
!find /kaggle/input/amazon-ml-2026-data/ -name "*.tsv" -exec cp {} data/ \;

# Overwrite config.yaml to point to this new data folder instead of C:/__MACOSX/
%%writefile config.yaml
data_dir: data
cache_dir: cache
output_dir: output
seed: 42
val_frac: 0.05
```

### Cell 3 (Optional): Configure Auto-Saving
If you want to use the script's automatic Kaggle Dataset backup feature:
```python
# Ensure you have KAGGLE_USERNAME and KAGGLE_KEY secrets set in the notebook Add-ons!
!mkdir -p /kaggle/working/ber-model-checkpoint
!kaggle datasets init -p /kaggle/working/ber-model-checkpoint

# Open /kaggle/working/ber-model-checkpoint/dataset-metadata.json 
# and replace the id and title with your own custom values.
```

### Cell 4: Train!
```bash
%%bash
# Execute the full training script!
bash run_kaggle.sh
```

---

## 5. Saving Your Progress
If you don't use Cell 3 to auto-sync to an external dataset, you can still easily save your run natively in Kaggle:
1. In the top right corner of the notebook, click **Save Version**.
2. Select **Save & Run All (Commit)**.
3. Click Save. 

Kaggle will run the entire notebook from top to bottom in the background. When it finishes, everything outputted to the `/kaggle/working/` folder (including your model checkpoints and parquet files) will be permanently zipped and saved to that notebook version's **Output** tab.
