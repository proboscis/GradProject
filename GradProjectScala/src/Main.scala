
import java.io.{ByteArrayInputStream, InputStream, File}
import java.nio.charset.Charset

import scala.sys.process._
import scala.io.Source
object Main{
  val ipython = "/usr/local/bin/ipython"

  def main(args: Array[String]) {
    implicit def stringIsFile(fn:String):File = new File(fn)
    def input(string:String) = new ByteArrayInputStream(Charset.forName("UTF-8").encode(string).array())
    def removeColorCode(str:String) = str.replaceAll("\u001B\\[[;\\d]*m", "")
    Process("python glyph/Main.py","../").lineStream.foreach(println)

    //val result = Process("ipython glyph/Main.py","../").!
    //println(result)
  }
}