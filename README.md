# Domain Generalized Category discovery (DGCD)

## Installation

Clone this repository

```bash
git clone https://github.com/Shubh-Nil/D_GCD.git
cd D_GCD
```

Set up the environment

```bash
conda create --name dgcd python==3.12
conda activate dgcd
pip install -r requirements.txt
```

## Data

Create a `datasets` directory and download the required datasets:

```bash
mkdir -p datasets
```

### Download PACS

Use the official Google Drive folder to download the PACS dataset:

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk -O datasets/PACS
```

([drive.google.com](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ&usp=sharing), [huggingface.co](https://huggingface.co/datasets/flwrlabs/pacs?utm_source=chatgpt.com))

### Download Office-Home

```bash
# Download and unzip the Office-Home dataset
pip install gdown
gdown 'https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg' -O datasets/OfficeHome.zip
unzip datasets/OfficeHome.zip -d datasets/OfficeHome
rm datasets/OfficeHome.zip
```

([github.com](https://github.com/LeoXinhaoLee/Imbalanced-Source-free-Domain-Adaptation))

### Download DomainNet

```bash
# Download and extract the cleaned DomainNet dataset
mkdir -p datasets/DomainNet
cd datasets/DomainNet
for domain in clipart infograph painting quickdraw real sketch; do
  echo "Downloading DomainNet ${domain}..."
  wget http://ai.bu.edu/M3SDA/${domain}.zip
  unzip ${domain}.zip -d ${domain}
  rm ${domain}.zip
done
```

([ai.bu.edu](https://ai.bu.edu/M3SDA/?utm_source=chatgpt.com))

## Synthetic Dataset 

python syn_data.py --dataset datasets/PACS --output datasets/PACS_synthetic
