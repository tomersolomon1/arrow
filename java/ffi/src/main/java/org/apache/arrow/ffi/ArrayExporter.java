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

import static org.apache.arrow.ffi.NativeUtil.NULL;
import static org.apache.arrow.ffi.NativeUtil.addressOrNull;

import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.ffi.jni.JniWrapper;
import org.apache.arrow.ffi.jni.PrivateData;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;

/**
 * Exporter for {@link ArrowArray}.
 */
final class ArrayExporter {
  /**
   * Private data structure for exported arrays.
   */
  static class ExportedArrayPrivateData implements PrivateData {
    ArrowBuf buffers_ptrs;
    List<ArrowBuf> buffers;
    ArrowBuf children_ptrs;
    List<ArrowArray> children;
    ArrowArray dictionary;

    @Override
    public void close() {
      NativeUtil.closeBuffer(buffers_ptrs);

      if (buffers != null) {
        for (ArrowBuf buffer : buffers) {
          NativeUtil.closeBuffer(buffer);
        }
      }
      NativeUtil.closeBuffer(children_ptrs);

      if (children != null) {
        for (ArrowArray child : children) {
          child.close();
        }
      }

      if (dictionary != null) {
        dictionary.close();
      }
    }
  }

  private final ArrowArray array;

  ArrayExporter(ArrowArray dst) {
    this.array = dst;
  }

  void export(BufferAllocator allocator, FieldVector vector, List<FieldVector> children) {
    export(allocator, vector.getFieldBuffers(), vector.getValueCount(), vector.getNullCount(), children);
  }

  void export(BufferAllocator allocator, List<ArrowBuf> buffers, int valueCount, int nullCount,
      List<FieldVector> children) {
    ExportedArrayPrivateData data = new ExportedArrayPrivateData();
    try {
      if (children != null) {
        data.children = new ArrayList<>(children.size());
        data.children_ptrs = allocator.buffer((long) children.size() * Long.BYTES);
        for (int i = 0; i < children.size(); i++) {
          ArrowArray child = ArrowArray.allocateNew(allocator);
          data.children.add(child);
          data.children_ptrs.writeLong(child.memoryAddress());
        }
      }

      if (buffers != null) {
        data.buffers = new ArrayList<>(buffers.size());
        data.buffers_ptrs = allocator.buffer((long) buffers.size() * Long.BYTES);
        for (ArrowBuf arrowBuf : buffers) {
          if (arrowBuf != null) {
            arrowBuf.getReferenceManager().retain();
            data.buffers_ptrs.writeLong(arrowBuf.memoryAddress());
          } else {
            data.buffers_ptrs.writeLong(NULL);
          }
          data.buffers.add(arrowBuf);
        }
      }

      ArrowArray.Snapshot snapshot = new ArrowArray.Snapshot();
      snapshot.length = valueCount;
      snapshot.null_count = nullCount;
      snapshot.offset = 0;
      snapshot.n_buffers = (data.buffers != null) ? data.buffers.size() : 0;
      snapshot.n_children = (data.children != null) ? data.children.size() : 0;
      snapshot.buffers = addressOrNull(data.buffers_ptrs);
      snapshot.children = addressOrNull(data.children_ptrs);
      snapshot.dictionary = NULL; // TODO: support dictionary export
      snapshot.release = NULL;
      array.save(snapshot);

      // sets release and private data
      JniWrapper.get().exportArray(this.array.memoryAddress(), data);
    } catch (Exception e) {
      data.close();
      throw e;
    }

    // Export children
    if (children != null) {
      for (int i = 0; i < children.size(); i++) {
        FieldVector childVector = children.get(i);
        ArrowArray child = data.children.get(i);
        ArrayExporter childExporter = new ArrayExporter(child);
        childExporter.export(allocator, childVector, childVector.getChildrenFromFields());
      }
    }
  }
}
