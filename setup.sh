export PYTHONPATH=".:${PYTHONPATH} "

# Setting up necessary python paths to link the modules of upsnet and cycada works
export PYTHONPATH="$(pwd)/UPSNet/:$(pwd)/:${PYTHONPATH}"

# Dependencies
# CYCADA
pip install -r requirements.txt

# UPSNET
cd UPSNet
./init.sh