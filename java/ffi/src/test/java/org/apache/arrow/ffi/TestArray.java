/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.arrow.ffi;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.junit.Test;

public class TestArray extends TestFFI {
  
  @Test
  public void testExportVectorSchemaRoot() throws IOException {
    String expected;
    VectorSchemaRoot imported;

    // Consumer allocates empty structures
    try (ArrowSchema consumerArrowSchema = ArrowSchema.allocateNew(rootAllocator());
        ArrowArray consumerArrowArray = ArrowArray.allocateNew(rootAllocator())) {

      try (VectorSchemaRoot vsr = createDummyVSR()) {
        expected = vsr.contentToTSVString();

        // Producer creates structures from exisitng memory pointers
        try (ArrowSchema arrowSchema = ArrowSchema.wrap(consumerArrowSchema.memoryAddress());
            ArrowArray arrowArray = ArrowArray.wrap(consumerArrowArray.memoryAddress())) {
          // Producer exports vector into the FFI structures
          FFI.exportVectorSchemaRoot(rootAllocator(), vsr, arrowArray, arrowSchema);
        }
      }

      // Consumer imports vector
      imported = FFI.importVectorSchemaRoot(rootAllocator(), consumerArrowSchema, consumerArrowArray);
    }

    // Ensure that imported VectorSchemaRoot is valid even after FFI structures
    // closed
    assertEquals(expected, imported.contentToTSVString());
    imported.close();
  }

  @Test
  public void testExportVector() throws IOException {
    String expected;

    // Consumer allocates empty structures
    try (ArrowSchema consumerArrowSchema = ArrowSchema.allocateNew(rootAllocator());
        ArrowArray consumerArrowArray = ArrowArray.allocateNew(rootAllocator())) {

      try (StructVector vector = createDummyStructVector()) {
        expected = vector.toString();

        // Producer creates structures from exisitng memory pointers
        try (ArrowSchema arrowSchema = ArrowSchema.wrap(consumerArrowSchema.memoryAddress());
            ArrowArray arrowArray = ArrowArray.wrap(consumerArrowArray.memoryAddress())) {
          // Producer exports vector into the FFI structures
          FFI.exportVector(rootAllocator(), vector, arrowArray, arrowSchema);
        }
      }

      // Consumer imports vector
      try (FieldVector vector = FFI.importVector(rootAllocator(), consumerArrowArray, consumerArrowSchema)) {
        assertEquals(expected, vector.toString());
      }
    }
  }
}
