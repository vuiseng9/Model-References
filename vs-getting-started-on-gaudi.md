

# Connect to AWS DL1 instance
Add the following to your ssh config. Do revise IdentityFile to point to your path. Note that ProxyCommand is mandatory if you are on Intel Network, also revise proxy program full path accordingly. If you set up as following, you can connect instance via vscode as well.
```
Host autox-gaudi
User ubuntu
Hostname ec2-35-86-150-67.us-west-2.compute.amazonaws.com
IdentityFile /path/to/autox-admin.pem
ProxyCommand "C:\Program Files\Git\mingw64\bin\connect.exe" -S proxy-dmz.intel.com:1080 %h %p
```

# Setup
We assume the instance is set up witn Habana Deep Learning Base AMI. The following steps are meant to get PyTorch up and running on Gaudi1 card.

### Set up Gaudi-enabled PyTorch
```bash
git clone https://github.com/HabanaAI/Setup_and_Install
cd Setup_and_Install/installation_scripts/PyTorch
./pytorch_installation.sh
```
Once installation is complete, use ```hl-smi``` to check Gaudi cards on the system, we should see 8 cards on DL1 instance.

### Validate PyTorch setup with Habana's example
```bash
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/examples/computer_vision/hello_world
python3 mnist.py --batch-size=64 --epochs=1 --lr=1.0 --gamma=0.7 --hpu
```
While the script is running, utilization of Gaudi card can be observed via ```hl-smi -l 1```

### Important notes
1. Most of the setup above is in user space, all python packages have been installed at local ```/home/ubuntu/.local/```. For any subsequent python package installation, please add ```--user``` to python installation, e.g.
    * ```pip install --user <package>```
    * ```pip install --user -r requirements.txt```
    * ```python3 setup.py develop --user```
2. For repository that installs pytorch out of the box, the Gaudi-enabled Pytorch will be overwritten. It is advisable to avoid overwriting by revising the installation of any repository. Optionally, ```./pytorch_installation.sh``` could re-install Gaudi-enabled pytorch.
3. Some packages might install pre-built program for example ninja, they will be install ```/home/ubuntu/.local/bin```. As this path is not part of the executable search path, some automated scripts may fail, revise search path ```export PATH=/home/ubuntu/.local/bin:${PATH}``` 

# Migration

For migration of existing PyTorch script to gaudi-enabled PyTorch, please see [official documentation](https://docs.habana.ai/en/latest/PyTorch/Migration_Guide/Porting_Simple_PyTorch_Model_to_Gaudi.html).
Few tips, always check ```hl-smi``` to ensure intended usage. ```HABANA_VISIBLE_DEVICES``` is the environment variable synonymous to ```CUDA_VISIBLE_DEVICES```.


# Custom Op Sample
```
# build
cd /vchua/Model-References/PyTorch/examples/custom_op/custom_relu
python3 setup.py build

# test
python3 hpu_custom_op_relu_test.py
```