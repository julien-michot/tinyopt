find_package(Sphinx)

if(Sphinx_FOUND)
    # TODO finish this
    add_sphinx_document(
        tinyopt-user-manual
        CONF_FILE "${CMAKE_CURRENT_LIST_DIR}/doc/user/conf.py"
        "${CMAKE_CURRENT_LIST_DIR}/doc/user/api.rst")
else()
    message(WARNING "Sphinx Missing")
endif()