# Tensorflow Clothing ML

This is a small ML application written in php with Tensorflow. It is designed to use ML to identify types of clothing.

Tutorial for this application can be found here: [https://www.tensorflow.org/tutorials/keras/classification](https://www.tensorflow.org/tutorials/keras/classification)

<br>

**Accuracy of trining**

<img src="https://raw.githubusercontent.com/jwoodrow99/tensorflow_clothing/main/results/training_accuracy.png"/>

<br><br>

**Results after training**

(0 = T-shirt/top, 1 = Trouser,  2 = Pullover, 3 = Dress, 4 = Coat, 5 = Sandal, 6 = Shirt, 7 = Sneaker, 8 = Bag, 9 = Ankle boot)

<br>

<img src="https://raw.githubusercontent.com/jwoodrow99/tensorflow_clothing/main/results/training_results.png"/>

<br><br>

## How to run

1. create virtual environment

    ``` bash
    python3 -m venv --system-site-packages ./tensor_flow_env
    ```

2. Enter VM

    ``` bash
    source ./tensor_flow_env/bin/activate
    ```

3. Clone repo

    ``` bash
    git clone https://github.com/jwoodrow99/tensorflow_clothing.git
    ```

4. Instal dependencies

    ``` bash
    cd tensorflow_clothing
    pip install -r requirements.txt
    ```

5. Run app

    ``` bash
    python app.py
    ```

## Notes

tensorflow does not work when installed with pipenv, useing venv. Dumped dependencies in requirements.txt file. to install dependencies run the following in venv:

``` bash
python3 -m pip install -r requirements.txt
```
