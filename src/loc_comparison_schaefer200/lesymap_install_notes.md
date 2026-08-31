# first had to brew install hdf5, cmake
wget -O ITKR.tar.gz https://github.com/ANTsX/ITKR/archive/refs/tags/v0.6.0.0.1.tar.gz
R CMD INSTALL ITKR.tar.gz
wget -O ANTsRCore.tar.gz https://github.com/ANTsX/ANTsRCore/archive/refs/tags/v0.8.0.tar.gz
R CMD INSTALL ANTsRCore.tar.gz
wget -O ANTsR.tar.gz https://github.com/ANTsX/ANTsR/archive/refs/tags/v0.6.0.tar.gz
R CMD INSTALL ANTsR.tar.gz
# had to install gfortran from here https://cran.r-project.org/bin/macosx/tools/
git clone https://github.com/dorianps/LESYMAP.git
R CMD INSTALL LESYMAP

# had to do these for the example
# www.xquartz.org download
install.packages("misc3d")
install.packages("pixmap")
library(misc3d)
library(pixmap)




