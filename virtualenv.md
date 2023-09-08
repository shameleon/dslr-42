# virtual environment

## Mac OS X

````
python3 -m venv venv
```

``` 
brew install pipx
pipx ensurepath
pipx install virtualenv
/Users/Astarte/Library/Python/3.11/bin/virtualenv -p python3 venv
./venv/bin install -r requirements.txt 
source ./venv/bin/activate
```
ps aux | grep background_script.py
kill PROCESS_ID