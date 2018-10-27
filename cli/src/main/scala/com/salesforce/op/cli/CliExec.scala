/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.cli

// scalastyle:off
// TODO(vlad): make sure that a simple intellij run fills in the resources
// @see https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/util/MetadataUtils.scala#L54
// scalastyle:on
import java.io.File

import com.salesforce.op.cli.gen.Ops
import org.apache.commons.io.FileUtils

class CliExec {
  protected val DEBUG = false

  private[cli] def delete(dir: File): Unit = {
    FileUtils.deleteDirectory(dir)
    if (dir.exists()) {
      throw new IllegalStateException(s"Directory '${dir.getAbsolutePath}' still exists")
    }
  }

  def main(args: Array[String]): Unit = try {
    val outcome = for {
      arguments <- CommandParser.parse(args, CliParameters())
      if arguments.command == "gen"
      settings <- arguments.values
    } yield Ops(settings).run()

    outcome getOrElse {
      CommandParser.showUsage()
      quit("wrong arguments", 1)
    }
  } catch {
    case x: Exception =>
      if (DEBUG) x.printStackTrace()
      val msg = Option(x.getMessage).getOrElse(x.getStackTrace.mkString("", "\n", "\n"))
      quit(msg)
  }

  def quit(errorMsg: String, code: Int = -1): Nothing = {
    System.err.println(errorMsg)
    sys.exit(code)
  }
}

object CLI {
  def main(args: Array[String]): Unit = (new CliExec).main(args)
}