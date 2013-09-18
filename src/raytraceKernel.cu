// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>

using glm::vec3;
using glm::cross;
using glm::length;
using glm::normalize;

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
//Thrust doc: https://developer.nvidia.com/thrust
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov)
{
  ray r;
  float width = resolution.x;
  float height = resolution.y;
  vec3 M = eye + view;
  vec3 A = cross(view, up);
  vec3 B = cross(A, view);
  vec3 H = (A * length(view) * tanf(fov.x * (PI/180.0f))) / length(A);
  vec3 V = -(B * length(view) * tanf(fov.y * (PI/180.0f))) / length(B); // LOOK: Multiplied by negative to flip the image
  vec3 P = M + ((2*x)/(width-1)-1)*H + ((2*y)/(height-1)-1)*V;
  vec3 D = P - eye;
  vec3 DN = glm::normalize(D);

  r.origin = P;
  r.direction = DN;
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* cudamat, int numberOfMat, material* cudalights, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if ((x<=resolution.x && y<=resolution.y))
  {
	  ray r = raycastFromCameraKernel(resolution, 0, x, y, cam.position, cam.view, cam.up, cam.fov);
	  vec3 isectPoint = vec3(0,0,0);
	  vec3 isectNormal = vec3(0,0,0);
	  float t = FLT_MAX;

	  // testing intersections
	  for (int i = 0 ; i < numberOfGeoms ; ++i)
	  {
		  if (geoms[i].type == GEOMTYPE::SPHERE)
		  {
			  // do cube intersection
			  vec3 isectPointTemp = vec3(0,0,0);
			  vec3 isectNormalTemp = vec3(0,0,0);

			  float dist = sphereIntersectionTest(geoms[i], r, isectPointTemp, isectNormalTemp);

			  if (dist < t && dist != -1)
			  {
				  t = dist;
				  isectPoint = isectPointTemp;
				  isectNormal = isectNormalTemp;
			  }
		  }
		  else if (geoms[i].type == GEOMTYPE::CUBE)
		  {
			  // do sphere intersection

		  }
		  else if (geoms[i].type == GEOMTYPE::MESH)
		  {
			  // do triangle intersections
		  }
	  }


	  // sphere intersection check
	  if (t != FLT_MAX)
	  {
		  colors[index] = vec3(1,0,0);
	  }
	  else
	  {
		  colors[index] = vec3(1,1,1);
	  }

	  // material check
	  //if (numberOfMat == 7)
		 // colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	  //else
		 // colors[index] = vec3(0,0,0);


	  // light check
	  //if (cudalights[0].emittance == 1 && cudalights[1].emittance == 15)
		 // colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	  //else
		 // colors[index] = vec3(0,0,0);
  }
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  // TODO: check number of lights and number of materials
  int numberOfLights = 0;
  int numberOfMat = 0;

  for (int i = 0 ; i < numberOfMaterials ; ++i)
  {
	 if (materials[i].emittance != 0)
		 numberOfLights++;
	 else
		 numberOfMat++;
  }
  
  // Regaring (void**) ...
  // All CUDA API functions return an error code (or cudaSuccess if no error occured). 
  // All other parameters are passed by reference. However, in plain C you cannot have references, 
  // that's why you have to pass an address of the variable that you want the return information to be stored. 
  // Since you are returning a pointer, you need to pass a double-pointer.
  material* cudamat = NULL;
  cudaMalloc((void**)&cudamat, numberOfMat * sizeof(material));
  cudaMemcpy(cudamat, materials, numberOfMat * sizeof(material), cudaMemcpyHostToDevice);

  material* cudalights = NULL;
  cudaMalloc((void**)&cudalights, numberOfLights * sizeof(material));
  cudaMemcpy(cudalights, &(materials[numberOfMat]), numberOfLights * sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
  //Q: Why don't we need to use cudaMalloc and cudaMemcpy here?
  //A: During kernel execution, the value cam will be put onto GPU memory stack since this value is not being passed by reference.
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;


  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamat, numberOfMat, cudalights, numberOfLights);

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
