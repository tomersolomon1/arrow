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

import os
import pyarrow as pa
import sys
import xml.etree.ElementTree as ET
import gc
import unittest
import jpype
import decimal


def setup_jvm():
    # This test requires Arrow Java to be built in the same source tree
    try:
        arrow_dir = os.environ["ARROW_SOURCE_DIR"]
    except KeyError:
        arrow_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
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


class TestPythonToJava(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.old_allocated_python = pa.total_allocated_bytes()
        self.allocator = jpype.JPackage("org").apache.arrow.memory.RootAllocator(sys.maxsize)

    def tearDown(self):
        self.allocator.close()
        gc.collect()
        diff_python = pa.total_allocated_bytes() - self.old_allocated_python
        self.assertEqual(
            pa.total_allocated_bytes(), self.old_allocated_python,
            f"PyArrow memory was not adequately released: {diff_python} bytes lost")
        self.allocator.close()

    def test_string_array(self):
        def string_array_generator():
            return pa.array([None, "a", "bb", "ccc"])
        self.round_trip_array(string_array_generator)

    def test_decimal_array(self):
        def decimal_array_generator():
            data = [
                round(decimal.Decimal(722.82), 2),
                round(decimal.Decimal(-934.11), 2),
                None,
            ]
            return pa.array(data, pa.decimal128(5, 2))
        self.round_trip_array(decimal_array_generator)

    def test_int_array(self):
        def int_array_generator():
            return pa.array([1, 2, 3], type=pa.int32())
        self.round_trip_array(int_array_generator)

    def test_list_array(self):
        def list_array_generator():
            return pa.array(
                [[], [0], [1, 2], [4, 5, 6]], pa.list_(pa.int64())
            )
        self.round_trip_array(list_array_generator)

    def test_struct_array(self):
        def struct_array_generator():
            fields = [
                ("f1", pa.int32()),
                ("f2", pa.string()),
            ]
            return pa.array(
                [
                    {"f1": 1, "f2": "a"},
                    None,
                    {"f1": 3, "f2": None},
                    {"f1": None, "f2": "d"},
                    {"f1": None, "f2": None},
                ],
                pa.struct(fields),
            )
        self.round_trip_array(struct_array_generator)

    def test_field(self):
        def bool_field_generator():
            pa.field("aa", pa.bool_())
        self.round_trip_field(bool_field_generator)

    def test_field_nested(self):
        def nested_field_generator():
            return pa.field("test", pa.list_(pa.int32()), nullable=True)
        self.round_trip_field(nested_field_generator)

    def test_field_metadata(self):
        def metadata_field_generator():
            return pa.field("aa", pa.bool_(), {"a": "b"})
        self.round_trip_field(metadata_field_generator)

    def round_trip_array(self, array_generator):
        original_arr = array_generator()
        arrow_array_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.allocateNew(self.allocator)
        arrow_schema_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(self.allocator)
        arrow_arr_ptr_pj = arrow_array_pj.memoryAddress()
        arrow_schema_ptr_pj = arrow_schema_pj.memoryAddress()
        py_value = original_arr.to_pylist()
        py_type = original_arr.type

        # Export from pyarrow
        print("Export from pyarrow")
        original_arr._export_to_c(arrow_arr_ptr_pj, arrow_schema_ptr_pj)
        del original_arr

        # Import into Java
        print("importing to java")
        vector = jpype.JPackage("org").apache.arrow.ffi.FFI.importVector(self.allocator, arrow_array_pj,
                                                                         arrow_schema_pj, None)

        # Export from Java
        print("Export from Java")
        arrow_array_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowArray.allocateNew(self.allocator)
        arrow_schema_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(self.allocator)
        arrow_arr_ptr_jp = arrow_array_jp.memoryAddress()
        arrow_schema_ptr_jp = arrow_schema_jp.memoryAddress()
        jpype.JPackage("org").apache.arrow.ffi.FFI.exportVector(self.allocator, vector, None, arrow_array_jp, arrow_schema_jp)

        # Import back to pyarrow
        print("Import back to pyarrow")
        arr_new = pa.Array._import_from_c(arrow_arr_ptr_jp, arrow_schema_ptr_jp)

        # Assert everything is fine
        assert arr_new.to_pylist() == py_value
        assert arr_new.type == py_type

        del arr_new

        # release java resources
        vector.close()
        arrow_array_pj.close()
        arrow_schema_pj.close()
        arrow_array_jp.close()
        arrow_schema_jp.close()
        print("end of round_trip_array")

    def round_trip_field(self, field_generator):
        field1 = field_generator()
        field2 = field_generator()
        arrow_schema_pj = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(self.allocator)

        # Export from pyarrow
        print("Export from pyarrow")
        arrow_schema_ptr_pj = arrow_schema_pj.memoryAddress()
        field1._export_to_c(arrow_schema_ptr_pj)
        del field1

        # Import to Java
        print("importing to java")
        jfield = jpype.JPackage("org").apache.arrow.ffi.FFI.importField(self.allocator, arrow_schema_pj, None)

        # Export from Java
        print("Export from Java")
        arrow_schema_jp = jpype.JPackage("org").apache.arrow.ffi.ArrowSchema.allocateNew(self.allocator)

        jpype.JPackage("org").apache.arrow.ffi.exportField(self.allocator, jfield, None, arrow_schema_jp)

        # Import to pyarrow
        print("Import back to pyarrow")
        arrow_schema_ptr_jp = arrow_schema_jp.memoryAddress()
        field3 = pa.Field._import_from_c(arrow_schema_ptr_jp)

        assert field2 == field3

        # release resources
        arrow_schema_pj.close()
        arrow_schema_jp.close()
        print("end of round_trip_field")


if __name__ == '__main__':
    setup_jvm()
    unittest.main(verbosity=2)
