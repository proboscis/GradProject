
import java.io.{ByteArrayInputStream, InputStream, File}
import java.nio.charset.Charset
import Util._
import scala.sys.process._
import scala.io.Source
import scalaz._
import Scalaz._
object Main{
  val ipython = "/usr/local/bin/ipython"
  implicit def stringIsFile(fn:String):File = new File(fn)
  def input(string:String) = new ByteArrayInputStream(Charset.forName("UTF-8").encode(string).array())
  def removeColorCode(str:String) = str.replaceAll("\u001B\\[[;\\d]*m", "")
  def main(args: Array[String]) {
    //Process("python glyph/Main.py","../").lineStream.foreach(println)
    """pwd""" !

    Process(Seq("python","ebooks.py"),"../glyph/")
      .lineStream_!.foreach(println)
    //val pathLine = """printenv""".lineStream_!.find(_.startsWith("PATH")).get
    //val paths = pathLine.replace("PATH=","").split(":").foreach(println)
    //val result = Process("ipython glyph/Main.py","../").!
    //println(result)

    val data = Seq("../glyph/env_in_idea.txt","../glyph/env_in_shell.txt")
    val sets@Seq(ideaSet,shellSet) = data.map(fileLines).map(_.toSet)
    val diff = (ideaSet ++ shellSet) &~ (ideaSet & shellSet)
    println(ideaSet.size)
    println(shellSet.size)
    println(diff.size)
    //diff foreach println
  }
}
