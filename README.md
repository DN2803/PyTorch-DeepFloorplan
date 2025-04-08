# PyTorch DeepFloorplan

## How to run?
1. Install packages. 
```
uv sync
```
2. Download the contents of the `annotations` directory [here](https://mycuhk-my.sharepoint.com/personal/1155052510_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155052510%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Ffloorplan%5Fmodel&ga=1) and put these into `/dataset`

3. Process the dataset:
```
uv run process_dataset.py
```
4. Train the network,
```
python main.py
```
5. Deploy the network, 
```
python deploy.py
```