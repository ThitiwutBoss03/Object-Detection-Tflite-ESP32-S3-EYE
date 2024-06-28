/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "object_detection_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

#include "esp_heap_caps.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
  constexpr int scratchBufSize = 150 * 1024;
#else
  constexpr int scratchBufSize = 0;
#endif
  // Increase tensor arena size
  constexpr int kTensorArenaSize = 100 * 1024 + scratchBufSize;

  static uint8_t* tensor_arena;

} // namespace


// The name of this function is important for Arduino compatibility.
void setup()
{
  model = tflite::GetModel(object_detection_model); // Updated to new model data
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

if (tensor_arena == NULL) {
  tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }
  else {
    printf("Allocated %d bytes for tensor arena\n", kTensorArenaSize);
  
}
}

  static tflite::MicroMutableOpResolver<16> micro_op_resolver;

  // Include only the necessary operations
  micro_op_resolver.AddConv2D();          // Conv2D layer
  micro_op_resolver.AddFullyConnected();  // Dense layer
  micro_op_resolver.AddMaxPool2D();       // MaxPooling2D layer
  micro_op_resolver.AddSoftmax();         // Softmax activation
  micro_op_resolver.AddQuantize();        // Quantize operation (if using quantized model)
  micro_op_resolver.AddDequantize();      // Dequantize operation (if using quantized model)
  micro_op_resolver.AddDepthwiseConv2D(); // DepthwiseConv2D layer
  micro_op_resolver.AddReshape();         // Reshape layer
  micro_op_resolver.AddAveragePool2D();   // AveragePooling2D layer
  // Add operations for BatchNormalization layers
  micro_op_resolver.AddMul();   // Used in BatchNormalization
  micro_op_resolver.AddAdd();   // Used in BatchNormalization
  micro_op_resolver.AddSub();   // Used in BatchNormalization
  micro_op_resolver.AddDiv();   // Used in BatchNormalization
  micro_op_resolver.AddMean();  // Used in BatchNormalization
  micro_op_resolver.AddRsqrt(); // Used in BatchNormalization

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk)
  {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
void loop()
{
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8))
  {
    MicroPrintf("Image capture failed.");
  }

  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor *output = interpreter->output(0);

  // Process the inference results.
  // TODO 1: Update the code to handle the 3 classes: cup, laptop, unknown ---



  // END TODO 1 ----------------------------------------------------------------

  RespondToDetection(cup_score_f, laptop_score_f, unknown_score_f);
  vTaskDelay(2); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
long long total_time = 0;
long long start_time = 0;
extern long long softmax_total_time;
extern long long dc_total_time;
extern long long conv_total_time;
extern long long fc_total_time;
extern long long pooling_total_time;
extern long long add_total_time;
extern long long mul_total_time;
#endif

void run_inference(void *ptr)
{
  for (int i = 0; i < kNumCols * kNumRows; i++)
  {
    input->data.int8[i] = ((uint8_t *)ptr)[i] ^ 0x80;
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  if (kTfLiteOk != interpreter->Invoke())
  {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor *output = interpreter->output(0);

  // Process the inference results.
  // TODO 1: Update the code to handle the 3 classes: cup, laptop, unknown ---


  // END TODO 1 ----------------------------------------------------------------


  RespondToDetection(cup_score_f, laptop_score_f, unknown_score_f);
}
