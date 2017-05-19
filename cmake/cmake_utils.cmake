cmake_minimum_required(VERSION 3.6)

function(add_src src_file)
	target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${src_file})
endfunction(add_src)