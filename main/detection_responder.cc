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

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "esp_main.h"
#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp32_s3_eye.h"
#include "esp_heap_caps.h"

// Camera definition is always initialized to match the trained detection model: 96x96 pix
// That is too small for LCD displays, so we extrapolate the image to 192x192 pix
#define IMG_WD (96 * 2)
#define IMG_HT (96 * 2)

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *status_indicator = NULL;
static lv_obj_t *label = NULL;
static lv_color_t *canvas_buf = NULL;

void check_memory_usage() {
    size_t free_heap_size = heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
    size_t free_spi_heap_size = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Free heap size: %zu bytes\n", free_heap_size);
    printf("Free SPI heap size: %zu bytes\n", free_spi_heap_size);
}

static void create_gui(void)
{
    bsp_display_cfg_t cfg = {
        .lvgl_port_cfg = {
            .task_priority = CONFIG_BSP_DISPLAY_LVGL_TASK_PRIORITY,
            .task_stack = 6144,
            .task_affinity = 1,
            .timer_period_ms = CONFIG_BSP_DISPLAY_LVGL_TICK,
        },
        .buffer_size = 240 * 20,
        .double_buffer = true,  // Use boolean instead of int
        .flags = {
            .buff_dma = true,
            .buff_spiram = false,
        }
    };

    bsp_display_start_with_config(&cfg);
    bsp_display_backlight_on(); // Set display brightness to 100%
    bsp_display_lock(0);

    // Check memory before allocation
    check_memory_usage();

    // Attempt to allocate memory for the canvas buffer
    if (canvas_buf == NULL) {
        canvas_buf = (lv_color_t *)heap_caps_malloc(IMG_WD * IMG_HT * sizeof(lv_color_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (canvas_buf == NULL) {
            printf("Failed to allocate SPI memory for canvas buffer\n");
            bsp_display_unlock();
            return;
        }
    }

    // Create and configure the camera canvas
    camera_canvas = lv_canvas_create(lv_scr_act());
    assert(camera_canvas);
    lv_canvas_set_buffer(camera_canvas, canvas_buf, IMG_WD, IMG_HT, LV_COLOR_FORMAT_NATIVE); 
    lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

    status_indicator = lv_led_create(lv_scr_act());
    assert(status_indicator);
    lv_obj_align(status_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);

    label = lv_label_create(lv_scr_act());
    assert(label);
    lv_label_set_text_static(label, "Status: Unknown");
    lv_obj_align_to(label, status_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);

    bsp_display_unlock();

    // Check memory after allocation
    check_memory_usage();
}

void RespondToDetection(float cup_score, float laptop_score, float unknown_score) {
    int cup_score_int = (cup_score) * 100 + 0.5;
    int laptop_score_int = (laptop_score) * 100 + 0.5;
    int unknown_score_int = (unknown_score) * 100 + 0.5;

#if DISPLAY_SUPPORT
    if (!camera_canvas) {
        create_gui();
        if (!camera_canvas) {
            printf("Failed to create GUI\n");
            return;
        }
    }

    uint16_t *buf = (uint16_t *) image_provider_get_display_buf();
    if (buf == NULL) {
        printf("Failed to get display buffer\n");
        return;
    }

    bsp_display_lock(0);
    
    // TODO: Determine the  Change the display to show different text and colors based on the detection result.
    

    // END TODO 1

    // Directly update the canvas buffer if it's allocated
    if (canvas_buf != NULL) {
        memcpy(canvas_buf, buf, IMG_WD * IMG_HT * sizeof(uint16_t));
        lv_obj_invalidate(camera_canvas);  // Invalidate the canvas to refresh
    } else {
        printf("Canvas buffer is NULL\n");
    }

    bsp_display_unlock();
#endif // DISPLAY_SUPPORT
    MicroPrintf("cup score:%d%%, laptop score:%d%%, unknown score:%d%%", cup_score_int, laptop_score_int, unknown_score_int);
}
#endif // DISPLAY_SUPPORT