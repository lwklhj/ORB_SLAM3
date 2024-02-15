echo "Configuring and building Thirdparty/libtorch ..."

cd Thirdparty
if [ ! -d libtorch ]; then
  wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu118.zip \
  && unzip libtorch-cxx11-abi-shared-with-deps-2.2.0+cu118.zip \
  && rm libtorch-cxx11-abi-shared-with-deps-2.2.0+cu118.zip
fi
cd ../

echo "Configuring and building Thirdparty/DBow3 ..."

cd Thirdparty/DBow3
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
