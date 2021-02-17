package org.apache.lucene.backward_codecs.lucene87;

import java.io.IOException;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;

public class Lucene87RWStoredFieldsFormat extends Lucene87StoredFieldsFormat {

  /** No-argument constructor. */
  public Lucene87RWStoredFieldsFormat() {
    super();
  }

  /** Constructor that takes a mode. */
  public Lucene87RWStoredFieldsFormat(Lucene87StoredFieldsFormat.Mode mode) {
    super(mode);
  }

  @Override
  public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo si, IOContext context)
      throws IOException {
    String previous = si.putAttribute(MODE_KEY, mode.name());
    if (previous != null && previous.equals(mode.name()) == false) {
      throw new IllegalStateException(
          "found existing value for "
              + MODE_KEY
              + " for segment: "
              + si.name
              + "old="
              + previous
              + ", new="
              + mode.name());
    }
    return impl(mode).fieldsWriter(directory, si, context);
  }
}
