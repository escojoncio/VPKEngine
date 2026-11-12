# vpgcn_lift_batch over a folder of oddly named files: names shorter than "<stage>_<hash>" must be skipped or
# reported, never abort the tool (it called substr(3, 18) on them and died with std::out_of_range).
file(REMOVE_RECURSE ${DIR})
file(MAKE_DIRECTORY ${DIR})
foreach(n vs.bin fs.bin cs.bin vs_.bin x.bin .fetch.bin vs.fetch.bin)
  file(WRITE ${DIR}/${n} "")
endforeach()
execute_process(COMMAND ${BATCH} ${DIR} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "vpgcn_lift_batch exited with ${rc}: ${err}")
endif()
if(NOT out MATCHES "programs:")
  message(FATAL_ERROR "vpgcn_lift_batch printed no summary: ${out}")
endif()
