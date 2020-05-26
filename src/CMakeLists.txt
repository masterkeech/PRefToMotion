add_library(PRefToMotion SHARED PRefToMotion.cpp)
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(PRefToMotion DDImage GLEW
            "-framework QuartzCore"
            "-framework IOKit"
            "-framework CoreFoundation"
            "-framework Carbon"
            "-framework ApplicationServices"
            "-framework OpenGL"
            "-framework AGL"
            )
elseif(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(PRefToMotion DDImage OpenGL)
endif ()

install(TARGETS PRefToMotion DESTINATION ${CMAKE_INSTALL_PREFIX})
