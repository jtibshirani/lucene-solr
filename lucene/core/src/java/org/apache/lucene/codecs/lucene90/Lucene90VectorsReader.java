/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.codecs.lucene90;

import java.io.IOException;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.PostingsReaderBase;
import org.apache.lucene.codecs.VectorValues;
import org.apache.lucene.codecs.VectorsReader;
import org.apache.lucene.codecs.lucene80.Lucene80DocValuesFormat;
import org.apache.lucene.codecs.lucene84.Lucene84PostingsFormat;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.Terms;

public class Lucene90VectorsReader extends VectorsReader {
  private final DocValuesProducer docValuesReader;
  private final FieldsProducer postingsReader;

  public Lucene90VectorsReader(SegmentReadState state) throws IOException {
    this.docValuesReader = new Lucene80DocValuesFormat().fieldsProducer(state);
    this.postingsReader = new Lucene84PostingsFormat().fieldsProducer(state);
  }

  @Override
  public void checkIntegrity() throws IOException {
    docValuesReader.checkIntegrity();
    postingsReader.checkIntegrity();
  }

  @Override
  public VectorValues getVectorValues(FieldInfo field) throws IOException {
    return new VectorValues() {
      @Override
      public BinaryDocValues getVectorValues() throws IOException {
        return docValuesReader.getBinary(field);
      }

      @Override
      public Terms getClusterPostings() throws IOException {
        return postingsReader.terms(field.name);
      }
    };
  }

  @Override
  public void close() throws IOException {
    docValuesReader.close();
    postingsReader.close();
  }
}
