Nerfstudio's command is just a wrapper, so it needs the actual COLMAP engine installed on your Linux machine to do the math. Since you are using Conda, this is incredibly easy:  
	conda install -c conda-forge colmap  
	git clone --recursive https://github.com/cvg/Hierarchical-Localization.git  
	cd Hierarchical-Localization  
	pip install -e .  
	cd ..  
	
To estimate the intrinsics and extrincs:  
	ns-process-data video --data data/demo1.mp4 --output-dir results/colmap_output --sfm-tool hloc --feature-type superpoint --matcher-type superpoint+lightglue
To train from the above genearted output:
	ns-train splatfacto --data results/colmap_output
Go to browser:
	http://localhost:7007

