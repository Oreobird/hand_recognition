idf_build_get_property(target IDF_TARGET)

set(srcs main.cpp
         hand_rec_model_coefficient.cpp
         app_camera.c
         app_httpd.c
    )

set(include_dirs ./)

idf_component_register(SRCS ${srcs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires})