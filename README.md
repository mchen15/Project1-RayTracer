-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
README:
-------------------------------------------------------------------------------
![alt tag](https://raw.github.com/mchen15/Project1-RayTracer/master/renders/2.png)

In this project, I have implemented a CUDA based raytracer utilizing the base code
from the course CIS565 at University of Pennsylvania. In the base code, I was provided
with functions such as file I/O, OpenGL setup, base CUDA kernel set up, etc. The following
are the features that I implemented. 

-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------
* Raycasting from a camera into a scene through a pixel grid
	- based on the camera configuration, a ray is cast through each pixel and intersected with
	  each object in the scene.
* Phong lighting for one point light source
	- Given an arbitrary number of light sources, Phong shading model is applied to each surface
	  attenuated with the distance away from the light source.
* Diffuse lambertian surfaces
* Raytraced shadows
	- Each light source will be used to cast shadows.
* Cube intersection testing
	- Ray-Cube intersection with slabs 
* Sphere surface point sampling
	- Randomly sampling a point on a sphere utilizing thrust.
* Supersampled antialiasing
	- Comparision:
	![alt tag](https://raw.github.com/mchen15/Project1-RayTracer/master/renders/ss%20comp.png)
* Specular highlights
* Simple mirror reflection
	- Rays are reflected from reflective surfaces and samples the color of the object that is
	  intersected.
* Soft shadows.

Work in Progress:
![alt tag](https://raw.github.com/mchen15/Project1-RayTracer/master/renders/1.png)

Video: http://youtu.be/Tnpnndr-x28

-------------------------------------------------------------------------------
Performance Evaluation
-------------------------------------------------------------------------------
Over the course of the project, I have attempted several experiments in order to speed up the 
run time of my program.

1. Dividing large kernels into smaller kernels.
	- While most of the core ray tracing algorithm is still in one kernel, I have taken out some of the 
	  more minor calculations and launched as separate kernels. Unfortunately, I did not notice a major
	  speed up as a result of this process.
	  
2. Changing the number of threads per block.
	- Since kernels issue instructs in warps with 32 threads, I decided to change the number of threads per
	  block to 32. There was a slight improvement in run time. With 8 threads per block, I was getting around
	  4 frames per second and with 32 threads per block, it went up to around 5.45 frames per second.
	
Ideas that I did not get to try.

1. Have each of the super sampling rays be separate threads as well.
2. Accumulate shadow ray colors across multiple iterations instead of casting multiple shadow rays per 
   iteration.