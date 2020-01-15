#! /bin/bash
source ~/.bash_profile



for i in {544111..544226}; do
    condor_rm -name llrt3condor.in2p3.fr $i
done


