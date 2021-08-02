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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.StructVectorLoader;
import org.apache.arrow.vector.StructVectorUnloader;
import org.apache.arrow.vector.VectorLoader;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.ArrowType.ArrowTypeID;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Functions for working with the C data interface.
 */
public final class FFI {
  /**
   * Export Java Field using the C data interface format.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param field     Field object to export
   * @param out       C struct where to export the field
   */
  public static void exportField(BufferAllocator allocator, Field field, ArrowSchema out) {
    SchemaExporter exporter = new SchemaExporter(out);
    exporter.export(allocator, field);
  }

  /**
   * Export Java Schema using the C data interface format.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param schema    Schema object to export
   * @param out       C struct where to export the field
   */
  public static void exportSchema(BufferAllocator allocator, Schema schema, ArrowSchema out) {
    // Convert to a struct field equivalent to the input schema
    FieldType fieldType = new FieldType(false, new ArrowType.Struct(), null, schema.getCustomMetadata());
    Field field = new Field("", fieldType, schema.getFields());
    exportField(allocator, field, out);
  }

  /**
   * Export Java FieldVector using the C data interface format.
   * <p>
   * The resulting ArrowArray struct keeps the array data and buffers alive until
   * its release callback is called by the consumer.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param vector    Vector object to export
   * @param out       C struct where to export the array
   */
  public static void exportVector(BufferAllocator allocator, FieldVector vector, ArrowArray out) {
    exportVector(allocator, vector, out, null);
  }

  /**
   * Export Java FieldVector using the C data interface format.
   * <p>
   * The resulting ArrowArray struct keeps the array data and buffers alive until
   * its release callback is called by the consumer.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param vector    Vector object to export
   * @param out       C struct where to export the array
   * @param outSchema Optional C struct where to export the array type
   */
  public static void exportVector(BufferAllocator allocator, FieldVector vector, ArrowArray out,
      ArrowSchema outSchema) {
    if (outSchema != null) {
      exportField(allocator, vector.getField(), outSchema);
    }

    ArrayExporter exporter = new ArrayExporter(out);
    exporter.export(allocator, vector, vector.getChildrenFromFields());
  }

  /**
   * Export the current contents of a Java VectorSchemaRoot using the C data
   * interface format.
   * <p>
   * The vector schema root is exported as if it were a struct array. The
   * resulting ArrowArray struct keeps the record batch data and buffers alive
   * until its release callback is called by the consumer.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param vsr       Vector schema root to export
   * @param out       C struct where to export the record batch
   */
  public static void exportVectorSchemaRoot(BufferAllocator allocator, VectorSchemaRoot vsr, ArrowArray out) {
    exportVectorSchemaRoot(allocator, vsr, out, null);
  }

  /**
   * Export the current contents of a Java VectorSchemaRoot using the C data
   * interface format.
   * <p>
   * The vector schema root is exported as if it were a struct array. The
   * resulting ArrowArray struct keeps the record batch data and buffers alive
   * until its release callback is called by the consumer.
   * 
   * @param allocator Buffer allocator for allocating C data interface fields
   * @param vsr       Vector schema root to export
   * @param out       C struct where to export the record batch
   * @param outSchema Optional C struct where to export the record batch schema
   */
  public static void exportVectorSchemaRoot(BufferAllocator allocator, VectorSchemaRoot vsr, ArrowArray out,
      ArrowSchema outSchema) {
    if (outSchema != null) {
      exportSchema(allocator, vsr.getSchema(), outSchema);
    }

    VectorUnloader unloader = new VectorUnloader(vsr);
    try (ArrowRecordBatch recordBatch = unloader.getRecordBatch()) {
      StructVectorLoader loader = new StructVectorLoader(vsr.getSchema());
      try (StructVector vector = loader.load(allocator, recordBatch)) {
        exportVector(allocator, vector, out);
      }
    }
  }

  /**
   * Import Java Schema from the C data interface.
   * <p>
   * The given ArrowSchema struct is released (as per the C data interface
   * specification), even if this function fails.
   * 
   * @param schema C data interface struct representing the field
   * @return Imported field object
   */
  public static Schema importSchema(ArrowSchema schema) {
    Field structField = importField(schema);
    if (structField.getType().getTypeID() != ArrowTypeID.Struct) {
      throw new IllegalArgumentException("Cannot import schema: ArrowSchema describes non-struct type");
    }
    return new Schema(structField.getChildren(), structField.getMetadata());
  }

  /**
   * Import Java Field from the C data interface.
   * <p>
   * The given ArrowSchema struct is released (as per the C data interface
   * specification), even if this function fails.
   * 
   * @param schema C data interface struct representing the field [inout]
   * @return Imported field object
   */
  public static Field importField(ArrowSchema schema) {
    try {
      SchemaImporter importer = new SchemaImporter();
      return importer.importField(schema);
    } finally {
      schema.release();
      schema.close();
    }
  }

  /**
   * Import Java vector from the C data interface.
   * <p>
   * The ArrowArray struct has its contents moved (as per the C data interface
   * specification) to a private object held alive by the resulting array.
   * 
   * @param allocator Buffer allocator
   * @param array     C data interface struct holding the array data
   * @param vector    Imported vector object [out]
   */
  public static void importIntoVector(BufferAllocator allocator, ArrowArray array, FieldVector vector) {
    ArrayImporter importer = new ArrayImporter(vector);
    importer.importArray(allocator, array);
  }

  /**
   * Import Java vector and its type from the C data interface.
   * <p>
   * The ArrowArray struct has its contents moved (as per the C data interface
   * specification) to a private object held alive by the resulting vector. The
   * ArrowSchema struct is released, even if this function fails.
   * 
   * @param allocator Buffer allocator for allocating the output FieldVector
   * @param array     C data interface struct holding the array data
   * @param schema    C data interface struct holding the array type
   * @return Imported vector object
   */
  public static FieldVector importVector(BufferAllocator allocator, ArrowArray array, ArrowSchema schema) {
    Field field = importField(schema);
    FieldVector vector = field.createVector(allocator);
    importIntoVector(allocator, array, vector);
    return vector;
  }

  /**
   * Import record batch from the C data interface into vector schema root.
   * 
   * The ArrowArray struct has its contents moved (as per the C data interface
   * specification) to a private object held alive by the resulting vector schema
   * root.
   * 
   * The schema of the vector schema root must match the input array (undefined
   * behavior otherwise).
   * 
   * @param allocator Buffer allocator
   * @param array     C data interface struct holding the record batch data
   * @param root      vector schema root to load into
   */
  public static void importIntoVectorSchemaRoot(BufferAllocator allocator, ArrowArray array, VectorSchemaRoot root) {
    try (StructVector structVector = StructVector.empty("", allocator)) {
      for (Field field : root.getSchema().getFields()) {
        structVector.addOrGet(field.getName(), field.getFieldType(), FieldVector.class);
      }
      importIntoVector(allocator, array, structVector);
      StructVectorUnloader unloader = new StructVectorUnloader(structVector);
      VectorLoader loader = new VectorLoader(root);
      try (ArrowRecordBatch recordBatch = unloader.getRecordBatch()) {
        loader.load(recordBatch);
      }
    }
  }

  /**
   * Import Java vector schema root from a C data interface Schema.
   * 
   * The type represented by the ArrowSchema struct must be a struct type array.
   * 
   * The ArrowSchema struct is released, even if this function fails.
   * 
   * @param allocator Buffer allocator for allocating the output VectorSchemaRoot
   * @param schema    C data interface struct holding the record batch schema
   * @return Imported vector schema root
   */
  public static VectorSchemaRoot importVectorSchemaRoot(BufferAllocator allocator, ArrowSchema schema) {
    return importVectorSchemaRoot(allocator, schema, null);
  }

  /**
   * Import Java vector schema root from the C data interface.
   * 
   * The type represented by the ArrowSchema struct must be a struct type array.
   * 
   * The ArrowArray struct has its contents moved (as per the C data interface
   * specification) to a private object held alive by the resulting record batch.
   * The ArrowSchema struct is released, even if this function fails.
   * 
   * Prefer {@link #importIntoVectorSchemaRoot} for loading array data while
   * reusing the same vector schema root.
   * 
   * @param allocator Buffer allocator for allocating the output VectorSchemaRoot
   * @param schema    C data interface struct holding the record batch schema
   * @param array     Optional C data interface struct holding the record batch
   *                  data
   * @return Imported vector schema root
   */
  public static VectorSchemaRoot importVectorSchemaRoot(BufferAllocator allocator, ArrowSchema schema,
      ArrowArray array) {
    VectorSchemaRoot vsr = VectorSchemaRoot.create(importSchema(schema), allocator);
    if (array != null) {
      importIntoVectorSchemaRoot(allocator, array, vsr);
    }
    return vsr;
  }
}
