echo "Copying theanorc to home folder"
cp .theanorc ~/
echo "PLEASE MAKSURE Directory amir is available in the home folder"
sudo apt-get -y install ipython
sudo apt-get -y install python-pandas python-xlsxwriter
echo "IPYTHON installed"
sudo apt-get -y install python-mako
echo "Mako installed"
#sudo -H pip install --upgrade pip
#echo "pip upgraded"
sudo -H pip install cython
echo "CYTHON with pip installed"
echo "Installing LIBGPUARRAY"
#LIBGPU ARRAY INSTALLATION

cd ../..
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
sudo make install
cd ..
sudo ldconfig
python setup.py build
sudo python setup.py install

echo "LIBGPUARRAY installed"
## End of lib GPU array
sudo -H pip install theano
echo "Theano Installed. Please run ipython and type import theano and check that there is no error"



