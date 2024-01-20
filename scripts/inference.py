#!/usr/bin/env python3

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()

def get_TRT(engine_file):
    return TRT(engine_file)

def load_trtengine(engine_file_path):
    print("Loading TRT engine from file :",engine_file_path)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

class TRT:
    
    def __init__(self, engine):
        
        self.engine = engine
        self.context =  self.engine.create_execution_context()
        self.input_gpu, self.output_cpu, self.output_gpu = self.allocate_buffer(self.engine)
    
    def allocate_buffer(self, engine):
        host_in_size = trt.volume(engine.get_binding_shape(0))
        host_out_size = trt.volume(engine.get_binding_shape(1))
        host_in_dtype = trt.nptype(engine.get_binding_dtype(0))
        host_out_dtype = trt.nptype(engine.get_binding_dtype(1))
        #input_cpu = np.ascontiguousarray(image)
        input_cpu = cuda.pagelocked_empty(host_in_size, host_in_dtype)
        output_cpu = cuda.pagelocked_empty(host_out_size, host_out_dtype)

        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        output_gpu = cuda.mem_alloc(output_cpu.nbytes)

        return input_gpu, output_cpu, output_gpu
    
    def predict(self, input):
        
            # Set input shape based on image dimensions for inference
            #context.set_binding_shape(engine.get_binding_index("input"), (1, 4, img_height, img_width))
        input = input.reshape(-1)    

        # Transfer input data to the GPU.
        cuda.memcpy_htod(self.input_gpu, input)
        # Run inference
        #self.context.execute_async(bindings=bindings, stream_handle=stream.handle)
        self.context.execute(1, [int(self.input_gpu), int(self.output_gpu)])
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh(self.output_cpu, self.output_gpu)
        # Synchronize the stream
        return self.output_cpu.reshape(self.output_gpu)