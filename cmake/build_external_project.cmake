function (build_external_project target prefix url branch)

	set(trigger_build_dir ${CMAKE_BINARY_DIR}/force_${target})

	#mktemp dir in build tree
	file(MAKE_DIRECTORY ${trigger_build_dir} ${trigger_build_dir}/build)

	#generate false dependency project
	set(CMAKE_LIST_CONTENT "
			cmake_minimum_required(VERSION 3.0)
			set(EXT_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
			if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
				set(EXT_CMAKE_BUILD_TYPE "Release")
			endif()
			project(ExternalProjectCustom)
			include(ExternalProject)
			ExternalProject_add(${target}
					PREFIX ${prefix}/extern/${target}
					GIT_REPOSITORY  ${url}
					GIT_TAG ${branch}
					CMAKE_ARGS ${ARGN} -DCMAKE_INSTALL_PREFIX:PATH=${prefix}/build/extern/${target}
			)
			add_custom_target(trigger_${target})
			add_dependencies(trigger_${target} ${target})
	")

	file(WRITE ${trigger_build_dir}/CMakeLists.txt "${CMAKE_LIST_CONTENT}")

	execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ..
			WORKING_DIRECTORY ${trigger_build_dir}/build
			)
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
			WORKING_DIRECTORY ${trigger_build_dir}/build
			)

endfunction()