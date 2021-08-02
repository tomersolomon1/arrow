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

import java.io.IOException;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;

public class TestSchema extends TestFFI {

  @Test
  public void testExportSchema() throws IOException {
    // Consumer allocates empty ArrowSchema
    try (ArrowSchema consumerArrowSchema = ArrowSchema.allocateNew(rootAllocator())) {
      try (VectorSchemaRoot vsr = createDummyVSR()) {
        // Producer creates ArrowSchema from exisitng memory pointer
        ArrowSchema arrowSchema = ArrowSchema.wrap(consumerArrowSchema.memoryAddress());
        FFI.exportSchema(rootAllocator(), vsr.getSchema(), arrowSchema);
        arrowSchema.close();
      }
      // Consumer releases the now loaded schema
      consumerArrowSchema.release();
    }
  }

  @Test
  public void testRoundtrip() throws IOException {
    String expected;

    // Consumer allocates empty ArrowSchema
    try (ArrowSchema consumerArrowSchema = ArrowSchema.allocateNew(rootAllocator())) {
      // Producer fills the schema with data
      try (VectorSchemaRoot vsr = createDummyVSR()) {
        expected = vsr.getSchema().toJson();
        try (ArrowSchema arrowSchema = ArrowSchema.wrap(consumerArrowSchema.memoryAddress())) {
          FFI.exportSchema(rootAllocator(), vsr.getSchema(), arrowSchema);
        }
      }

      // Consumer imports schema
      Schema schema = FFI.importSchema(consumerArrowSchema);
      Assertions.assertEquals(expected, schema.toJson());
    }
  }
}
