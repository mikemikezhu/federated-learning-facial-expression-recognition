 # Install nbconvert package
 pip install --upgrade nbconvert

 # Remove previously generated README.md file
 rm -rf federated_learning_files/
 rm -rf README.md

 # Convert jupyter notebook to markdown
 jupyter nbconvert --to markdown federated_learning.ipynb

 # Rename README.md
 mv federated_learning.md README.md