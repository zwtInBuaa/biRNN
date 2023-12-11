# Description
The source codes of RITS-I, RITS, BRITS-I, BRITS for health-care data imputation/classification

To run the code:

python main.py --epochs 1000 --batch_size 32 --model brits

# Data Format

The data format is as follows:

* Each line in json/json is a string represents a python dict
* The structure of each dict is
    * forward
    * backward
    * label #对模型没用

    'forward' and 'backward' is a list of python dicts, which represents the input sequence in forward/backward directions. As an example for forward direction, each dict in the sequence contains:
    * values: list, indicating x_t \in R^d (after elimination)
    * masks: list, indicating m_t \in R^d
    * deltas: list, indicating \delta_t \in R^d
    * forwards: list, the forward imputation, only used in GRU_D, can be any numbers in our model #对模型没用
    * evals: list, indicating x_t \in R^d (before elimination)
    * eval_masks: list, indicating whether each value is an imputation ground-truth


首先运行slovete.py进行数据的处理，此文件里面有个参数n代表连续时间步长度。
要训练模型：
python main.py --epochs 1000 --batch_size 32 --model brits
这3个参数可以改变epochs，batch_size 和model
