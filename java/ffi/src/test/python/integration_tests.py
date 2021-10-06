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

import decimal
import gc
import os
import sys
import unittest
import xml.etree.ElementTree as ET

import jpype
import pyarrow as pa
from pyarrow.cffi import ffi


def setup_jvm():
    # This test requires Arrow Java to be built in the same source tree
    try:
        arrow_dir = os.environ["ARROW_SOURCE_DIR"]
    except KeyError:
        arrow_dir = os.path.join(os.path.dirname(
            __file__), '..', '..', '..', '..')
    pom_path = os.path.join(arrow_dir, '../java', 'pom.xml')
    tree = ET.parse(pom_path)
    version = tree.getroot().find(
        'POM:version',
        namespaces={
            'POM': 'http://maven.apache.org/POM/4.0.0'
        }).text
    jar_path = os.path.join(
        arrow_dir, '../java', 'tools', 'target',
        'arrow-tools-{}-jar-with-dependencies.jar'.format(version))
    jar_path = os.getenv("ARROW_TOOLS_JAR", jar_path)
    jar_path += ":{}".format(os.path.join(arrow_dir,
                                          "../java", "ffi/target/arrow-ffi-{}.jar".format(version)))
    kwargs = {}
    # This will be the default behaviour in jpype 0.8+
    kwargs['convertStrings'] = False
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Djava.class.path=" + jar_path, **kwargs)


class UuidType(pa.PyExtensionType):
    def __init__(self):
        super().__init__(pa.binary(16))

    def __reduce__(self):
        return UuidType, ()


class Bridge:
    def __init__(self) -> None:
        self.allocator = jpype.JPackage(
            "org").apache.arrow.memory.RootAllocator(sys.maxsize)
        self.jffi = jpype.JPackage("org").apache.arrow.ffi

    def java_to_python_field(self, jfield):
        c_schema = ffi.new("struct ArrowSchema*")
        ptr_schema = int(ffi.cast("uintptr_t", c_schema))
        self.jffi.FFI.exportField(self.allocator, jfield, None,
                                  self.jffi.ArrowSchema.wrap(ptr_schema))
        return pa.Field._import_from_c(ptr_schema)

    def java_to_python_array(self, vector, dictionary_provider=None):
        c_schema = ffi.new("struct ArrowSchema*")
        ptr_schema = int(ffi.cast("uintptr_t", c_schema))
        c_array = ffi.new("struct ArrowArray*")
        ptr_array = int(ffi.cast("uintptr_t", c_array))
        self.jffi.FFI.exportVector(self.allocator, vector, dictionary_provider, self.jffi.ArrowArray.wrap(
            ptr_array), self.jffi.ArrowSchema.wrap(ptr_schema))
        return pa.Array._import_from_c(ptr_array, ptr_schema)

    def java_to_python_record_batch(self, root):
        c_schema = ffi.new("struct ArrowSchema*")
        ptr_schema = int(ffi.cast("uintptr_t", c_schema))
        c_array = ffi.new("struct ArrowArray*")
        ptr_array = int(ffi.cast("uintptr_t", c_array))
        self.jffi.FFI.exportVectorSchemaRoot(self.allocator, root, None, self.jffi.ArrowArray.wrap(
            ptr_array), self.jffi.ArrowSchema.wrap(ptr_schema))
        return pa.RecordBatch._import_from_c(ptr_array, ptr_schema)

    def python_to_java_field(self, field):
        c_schema = self.jffi.ArrowSchema.allocateNew(self.allocator)
        field._export_to_c(c_schema.memoryAddress())
        return self.jffi.FFI.importField(self.allocator, c_schema, None)

    def python_to_java_array(self, array, dictionary_provider=None):
        c_schema = self.jffi.ArrowSchema.allocateNew(self.allocator)
        c_array = self.jffi.ArrowArray.allocateNew(self.allocator)
        array._export_to_c(c_array.memoryAddress(), c_schema.memoryAddress())
        return self.jffi.FFI.importVector(self.allocator, c_array, c_schema, dictionary_provider)

    def python_to_java_record_batch(self, record_batch):
        c_schema = self.jffi.ArrowSchema.allocateNew(self.allocator)
        c_array = self.jffi.ArrowArray.allocateNew(self.allocator)
        record_batch._export_to_c(
            c_array.memoryAddress(), c_schema.memoryAddress())
        # TODO: swap array and schema in Java API for consistency
        return self.jffi.FFI.importVectorSchemaRoot(self.allocator, c_schema, c_array, None)

    def close(self):
        self.allocator.close()


class TestPythonIntegration(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.old_allocated_python = pa.total_allocated_bytes()
        self.bridge = Bridge()

    def tearDown(self):
        self.bridge.close()
        gc.collect()
        diff_python = pa.total_allocated_bytes() - self.old_allocated_python
        self.assertEqual(
            pa.total_allocated_bytes(), self.old_allocated_python,
            f"PyArrow memory was not adequately released: {diff_python} bytes lost")

    def round_trip_field(self, field_generator):
        original_field = field_generator()
        java_field = self.bridge.python_to_java_field(original_field)
        del original_field
        new_field = self.bridge.java_to_python_field(java_field)
        del java_field

        expected = field_generator()
        self.assertEqual(expected, new_field)

    def round_trip_array(self, array_generator, expected_diff=None):
        original_arr = array_generator()
        with self.bridge.jffi.FFIDictionaryProvider() as dictionary_provider,\
                self.bridge.python_to_java_array(original_arr, dictionary_provider) as vector:
            del original_arr
            new_array = self.bridge.java_to_python_array(vector, dictionary_provider)

        expected = array_generator()
        if expected_diff:
            self.assertEqual(expected, new_array.view(expected.type))
        self.assertEqual(expected.diff(new_array), expected_diff or '')
        if dictionary_provider:
            dictionary_provider.close()
        # self.assertEqual(expected, new_array, expected.diff(new_array))

    def round_trip_record_batch(self, rb_generator):
        original_rb = rb_generator()
        with self.bridge.python_to_java_record_batch(original_rb) as root:
            del original_rb
            new_rb = self.bridge.java_to_python_record_batch(root)

        expected = rb_generator()
        self.assertEqual(expected, new_rb)

    def test_string_array(self):
        self.round_trip_array(lambda: pa.array([None, "a", "bb", "ccc"]))

    def test_decimal_array(self):
        data = [
            round(decimal.Decimal(722.82), 2),
            round(decimal.Decimal(-934.11), 2),
            None,
        ]
        self.round_trip_array(lambda: pa.array(data, pa.decimal128(5, 2)))

    def test_int_array(self):
        self.round_trip_array(lambda: pa.array([1, 2, 3], type=pa.int32()))

    def test_list_array(self):
        self.round_trip_array(lambda: pa.array(
            [[], [0], [1, 2], [4, 5, 6]], pa.list_(pa.int64())
        ), "# Array types differed: list<item: int64> vs list<$data$: int64>\n")

    def test_struct_array(self):
        fields = [
            ("f1", pa.int32()),
            ("f2", pa.string()),
        ]
        data = [
            {"f1": 1, "f2": "a"},
            None,
            {"f1": 3, "f2": None},
            {"f1": None, "f2": "d"},
            {"f1": None, "f2": None},
        ]
        self.round_trip_array(lambda: pa.array(data, type=pa.struct(fields)))

    @unittest.skip("bug in union")
    def test_sparse_union_array(self):
        types = pa.array([0, 1, 1, 0, 1], pa.int8())
        data = [
            pa.array(["a", "", "", "", "c"], pa.utf8()),
            pa.array([0, 1, 2, None, 0], pa.int64()),
        ]
        self.round_trip_array(
            lambda: pa.UnionArray.from_sparse(types, data))

    @unittest.skip("bug in union")
    def test_dense_union_array(self):
        types = pa.array([0, 1, 1, 0, 1], pa.int8())
        offsets = pa.array([0, 1, 2, 3, 4], type=pa.int32())
        data = [
            pa.array(["a", "", "", "", "c"], pa.utf8()),
            pa.array([0, 1, 2, None, 0], pa.int64()),
        ]
        self.round_trip_array(
            lambda: pa.UnionArray.from_dense(types, offsets, data))

    def test_dict(self):
        self.round_trip_array(
            lambda: pa.array(["a", "b", None, "d"], pa.dictionary(pa.int64(), pa.utf8())))

    def test_map(self):
        offsets = [0, None, 2, 6]
        pykeys = [b"a", b"b", b"c", b"d", b"e", b"f"]
        pyitems = [1, 2, 3, None, 4, 5]
        keys = pa.array(pykeys, type="binary")
        items = pa.array(pyitems, type="i4")
        self.round_trip_array(
            lambda: pa.MapArray.from_arrays(offsets, keys, items))

    def test_field(self):
        self.round_trip_field(lambda: pa.field("aa", pa.bool_()))

    def test_field_nested(self):
        self.round_trip_field(lambda: pa.field(
            "test", pa.list_(pa.int32()), nullable=True))

    def test_field_metadata(self):
        self.round_trip_field(lambda: pa.field("aa", pa.bool_(), {"a": "b"}))

    @unittest.skip("not supported by the latest released version yet")
    def test_field_extension(self):
        self.round_trip_field(lambda: pa.field("aa", UuidType()))

    def test_record_batch_with_list(self):
        data = [
            pa.array([[1], [2], [3], [4, 5, 6]]),
            pa.array([1, 2, 3, 4]),
            pa.array(['foo', 'bar', 'baz', None]),
            pa.array([True, None, False, True])
        ]
        self.round_trip_record_batch(
            lambda: pa.RecordBatch.from_arrays(data, ['f0', 'f1', 'f2', 'f3']))


if __name__ == '__main__':
    setup_jvm()
    unittest.main(verbosity=2)