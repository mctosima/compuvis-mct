def downloaddata():
    json_file_kaggle = json.load(open("./data/flickr8k/kaggle.json"))
    os.environ['KAGGLE_USERNAME'] = json_file_kaggle['username']
    os.environ['KAGGLE_KEY'] = json_file_kaggle['key']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('adityajn105/flickr8k', path="./data/flickr8k")
    print("Download Complete")

    from zipfile import ZipFile
    file_name = "./data/flickr8k/flickr8k.zip"
    with ZipFile(file_name, 'r') as zip:
        print('Extracting all the files now...')
        zip.extractall(path="./data/flickr8k")
        print('Done!')

def main():
    downloaddata()

if __name__ == '__main__':
    main()