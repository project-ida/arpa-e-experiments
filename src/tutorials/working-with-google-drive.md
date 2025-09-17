---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/arpa-e-experiments/blob/main/tutorials/working-with-google-drive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/arpa-e-experiments/blob/main/tutorials/working-with-google-drive.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Working with data stored in Google Drive


## Downloading data


Some of the data we work with is stored in a publicly viewable Google drive folder.

Because it is publicly viewable, we can use a simple library called [gdown](https://github.com/wkentaro/gdown) to pull the data into a notebook.

```python
import gdown
```

### Downloading a folder


Let's download a folder called `Sandbox` that we use for testing.

Once you grab the sharing link for that folder we just plug it striaght into `gdown`

```python
gdown.download_folder("https://drive.google.com/drive/folders/1WXlEJHSRLz2ailnHu9G9qorSogrW1PSs?usp=drive_link")
```

You should now be able to see a new folder called `Sandbox` that's sitting next to this notebook. The files inside should be the same as those found in Google drive.


### Downloading a file


Let's just download the `README.md` file instead of the whole folder.

If you use the sharing link for that specific file then you'll need to use the `fuzzy=True` option soo that gdown can extract the file ID more reliably.

```python
gdown.download("https://drive.google.com/file/d/1Ar48NIOHTP-PMg-4fg_VyjZHJbwrrnao/view?usp=drive_link", fuzzy=True)
```

You should now find `README.txt` sitting next to this notebook.


## Uploading data


Although our Google drive folder is publicly viewable, it's **NOT** publicly editable. This means that, in general, if you generate files in a notebook like this you'll need to do some form of authentication before you can upload files to the drive.

It's really complicated to do this from a notebook that's running locally on your computer, so if you need to work locally then it's simplest just to drag those files into Google drive manually.

If that's not an option, then you can work with the notebook inside of Google Colab because you're already authenticated inside that environment. 

However, the Drive integration with Colab doesn't let you use a sharing link to identify the folder in which you want to save the file. You must instead identify the folder based on how you've personally added the shared folder to your own drive. Different people might have added the shared folder in different locations in their  drives which makes it impossible to make code that will work for everyone.

Below is an example of what that could look like, but I **strongly** encourage you to opt for manual upload.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Create a new file and save data directly to it
with open('/content/drive/My Drive/MIT ARPA-E LENR folder/new_file.txt', 'w') as file:
    file.write("Content to save.")
```

```python
# Move an existing file into the Google drive folder

import shutil

# Replace with your shared drive folder path and filename
shared_folder_path = '/content/drive/My Drive/MIT ARPA-E LENR folder'
filename = 'existing_file.txt'

shutil.move(filename, f'{shared_folder_path}/{filename}')
```
