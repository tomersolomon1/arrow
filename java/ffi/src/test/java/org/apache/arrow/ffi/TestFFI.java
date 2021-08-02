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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.Types.MinorType;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.ArrowType.Struct;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.junit.After;
import org.junit.Before;

public abstract class TestFFI {
  private RootAllocator allocator = null;

  @Before
  public void setUp() {
    allocator = new RootAllocator(Long.MAX_VALUE);
  }

  @After
  public void tearDown() {
    try {
      allocator.close();
    } catch (IllegalStateException e) {
      e.printStackTrace();
    }
  }

  protected RootAllocator rootAllocator() {
    return allocator;
  }

  protected StructVector createDummyStructVector() {
    FieldType type = new FieldType(true, Struct.INSTANCE, null, null);
    StructVector vector = new StructVector("struct", allocator, type, null);
    IntVector a = vector.addOrGet("a", FieldType.nullable(MinorType.INT.getType()), IntVector.class);
    IntVector b = vector.addOrGet("b", FieldType.nullable(MinorType.INT.getType()), IntVector.class);
    IntVector c = vector.addOrGet("c", FieldType.nullable(MinorType.INT.getType()), IntVector.class);
    IntVector d = vector.addOrGet("d", FieldType.nullable(MinorType.INT.getType()), IntVector.class);
    for (int j = 0; j < 5; j++) {
      a.setSafe(j, j);
      b.setSafe(j, j);
      c.setSafe(j, j);
      d.setSafe(j, j);
      vector.setIndexDefined(j);
    }
    a.setValueCount(5);
    b.setValueCount(5);
    c.setValueCount(5);
    d.setValueCount(5);
    vector.setValueCount(5);
    return vector;
  }

  protected VectorSchemaRoot createDummyVSR() {

    Map<String, String> metadata = new HashMap<>();
    metadata.put("key", "value");
    IntVector a = new IntVector("a", new FieldType(true, new ArrowType.Int(32, true), null, metadata), allocator);
    IntVector b = new IntVector("b", allocator);
    IntVector c = new IntVector("c", allocator);
    IntVector d = new IntVector("d", allocator);

    for (int j = 0; j < 5; j++) {
      a.setSafe(j, j);
      b.setSafe(j, j);
      c.setSafe(j, j);
      d.setSafe(j, j);
    }
    a.setValueCount(5);
    b.setValueCount(5);
    c.setValueCount(5);
    d.setValueCount(5);

    List<Field> fields = Arrays.asList(a.getField(), b.getField(), c.getField(), d.getField());
    List<FieldVector> vectors = Arrays.asList(a, b, c, d);
    VectorSchemaRoot vsr = new VectorSchemaRoot(fields, vectors);

    return vsr;
  }
}
