
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
        "detection_responder.cc"
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "model_settings.cc"
        "object_detection_model_data.cc"
        "app_camera_esp.c"
        "esp_cli.c"

    PRIV_REQUIRES console spi_flash
    INCLUDE_DIRS "")
