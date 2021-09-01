# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import os
import pyarrow as pa
import pyarrow.jvm as pa_jvm
import pytest
import sys
import xml.etree.ElementTree as ET

import gc

try:
    from pyarrow.cffi import ffi
except ImportError:
    ffi = None

import pytest
# jpype = pytest.importorskip("jpype")
import jpype

try:
    import pandas as pd
    import pandas.testing as tm
except ImportError:
    pd = tm = None

needs_cffi = pytest.mark.skipif(ffi is None,
                                reason="test needs cffi package installed")

assert_schema_released = pytest.raises(
    ValueError, match="Cannot import released ArrowSchema")

assert_array_released = pytest.raises(
    ValueError, match="Cannot import released ArrowArray")

assert_stream_released = pytest.raises(
    ValueError, match="Cannot import released ArrowArrayStream")


def get_root_allocator():
    # This test requires Arrow Java to be built in the same source tree
    try:
        arrow_dir = os.environ["ARROW_SOURCE_DIR"]
    except KeyError:
        arrow_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')
    pom_path = os.path.join(arrow_dir, 'java', 'pom.xml')
    print(pom_path)
    tree = ET.parse(pom_path)
    version = tree.getroot().find(
        'POM:version',
        namespaces={
            'POM': 'http://maven.apache.org/POM/4.0.0'
        }).text
    jar_path = os.path.join(
        arrow_dir, 'java', 'tools', 'target',
        'arrow-tools-{}-jar-with-dependencies.jar'.format(version))
    jar_path = os.getenv("ARROW_TOOLS_JAR", jar_path)

    # Manual for now
    jar_path += ":{}".format(os.path.join(arrow_dir, "java", "ffi/target/arrow-ffi-6.0.0-SNAPSHOT.jar"))
    print(jar_path)
    kwargs = {}
    # This will be the default behaviour in jpype 0.8+
    kwargs['convertStrings'] = False
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + jar_path, "-Xint", "-Xdebug", "-Xnoagent",
                   "-Xrunjdwp:transport=dt_socket,server=y,address=12999,suspend=n",
                   **kwargs)
    return jpype.JPackage("org").apache.arrow.memory.RootAllocator(sys.maxsize)


@needs_cffi
def test_pjp_array_1(root_allocator):
    arrow_array_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.allocateNew(root_allocator)
    arrow_schema_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(root_allocator)
    arrow_arr_ptr_pj = arrow_array_pj.memoryAddress()
    arrow_schema_ptr_pj = arrow_schema_pj.memoryAddress()

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    arr = pa.array([1, 2, 3], type=pa.int32())
    py_value = arr.to_pylist()

    # Export from pyarrow
    print("Export from pyarrow")
    arr._export_to_c(arrow_arr_ptr_pj, arrow_schema_ptr_pj)
    del arr

    # Import into Java
    print("importing to java")
    vector = jpype.JPackage("org").apache.arrow.ffi.FFI.importVector(root_allocator, arrow_array_pj, arrow_schema_pj)

    # Export from Java
    print("Export from Java")
    arrow_array_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.allocateNew(root_allocator)
    arrow_schema_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(root_allocator)
    arrow_arr_ptr_jp = arrow_array_jp.memoryAddress()
    arrow_schema_ptr_jp = arrow_schema_jp.memoryAddress()
    jpype.JPackage("org").apache.arrow.ffi.FFI.exportVector(root_allocator, vector, arrow_array_jp, arrow_schema_jp)

    # Import back to pyarrow
    print("Import back to pyarrow")
    arr_new = pa.Array._import_from_c(arrow_arr_ptr_jp, arrow_schema_ptr_jp)
    # Assert everything is fine
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.int32()
    assert pa.total_allocated_bytes() > old_allocated

    del arr_new
    del vector
    del arrow_array_pj
    del arrow_schema_pj
    del arrow_array_jp
    del arrow_schema_jp
    del root_allocator

    assert pa.total_allocated_bytes() == old_allocated  # doesn't pass yet

    # Now released
    with assert_schema_released:
        pa.Array._import_from_c(arrow_arr_ptr_pj, arrow_schema_ptr_pj)
        pa.Array._import_from_c(arrow_arr_ptr_jp, arrow_schema_ptr_jp)
    print("end of test_pjp_array")


@needs_cffi
def test_pjp_array_2(root_allocator):
    # Allocating python -> java structs
    c_schema_pj = ffi.new("struct ArrowSchema*")
    ptr_schema_pj = int(ffi.cast("uintptr_t", c_schema_pj))
    c_array_pj = ffi.new("struct ArrowArray*")
    ptr_array_pj = int(ffi.cast("uintptr_t", c_array_pj))

    # Allocating Java -> Python structs
    c_schema_jp = ffi.new("struct ArrowSchema*")
    ptr_schema_jp = int(ffi.cast("uintptr_t", c_schema_jp))
    c_array_jp = ffi.new("struct ArrowArray*")
    ptr_array_jp = int(ffi.cast("uintptr_t", c_array_jp))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    arr = pa.array([1, 2, 3], type=pa.int32())
    py_value = arr.to_pylist()

    # Export from pyarrow
    print("Export from pyarrow")
    arr._export_to_c(ptr_array_pj, ptr_schema_pj)
    del arr

    # Import into Java
    print("importing to java")
    arrow_array_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.wrap(ptr_array_pj)
    arrow_schema_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.wrap(ptr_schema_pj)
    print("managed wrapping, now importing")
    vector = jpype.JPackage("org").apache.arrow.ffi.FFI.importVector(root_allocator, arrow_array_pj, arrow_schema_pj)

    # Export from Java
    print("Export from Java")
    arrow_array_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.wrap(ptr_array_jp)
    arrow_schema_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.wrap(ptr_schema_jp)
    jpype.JPackage("org").apache.arrow.ffi.FFI.exportVector(root_allocator, vector, arrow_array_jp, arrow_schema_jp)

    # Import back to pyarrow
    print("Import back to pyarrow")
    arr_new = pa.Array._import_from_c(ptr_array_jp, ptr_schema_jp)

    # Assert everything is fine
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.int32()
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new
    del vector
    del c_schema_pj
    del c_array_pj
    del c_schema_jp
    del c_array_jp

    del arrow_array_pj
    del arrow_schema_pj
    del arrow_array_jp
    del arrow_schema_jp

    del root_allocator
    assert pa.total_allocated_bytes() == old_allocated  # doesn't pass yet
    # Now released
    print("passed checks, releasing")
    with assert_schema_released:
        pa.Array._import_from_c(ptr_array_pj, ptr_schema_pj)
        pa.Array._import_from_c(ptr_array_jp, ptr_schema_jp)


# The original test from test_cffi
def test_3():
    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))
    c_array = ffi.new("struct ArrowArray*")
    ptr_array = int(ffi.cast("uintptr_t", c_array))

    gc.collect()  # Make sure no Arrow data dangles in a ref cycle
    old_allocated = pa.total_allocated_bytes()

    # Type is exported and imported at the same time
    arr = pa.array([1, 2, 3], type=pa.int32())
    py_value = arr.to_pylist()
    arr._export_to_c(ptr_array, ptr_schema)
    # Delete and recreate C++ objects from exported pointers
    del arr
    arr_new = pa.Array._import_from_c(ptr_array, ptr_schema)
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.int32()
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new
    assert pa.total_allocated_bytes() == old_allocated
    # Now released
    with assert_schema_released:
        pa.Array._import_from_c(ptr_array, ptr_schema)


if __name__ == "__main__":
    allocator = get_root_allocator()
    print("created allocator")
    test_pjp_array_2(allocator)
    test_3()
