/**
 * @author kentomasui
 */

import java.io._
import java.net.URL

import net.coobird.thumbnailator.Thumbnails
import net.coobird.thumbnailator.resizers.configurations.ScalingMode

import scala.util.Try
import scalaz._
import Scalaz._
import scalaz.effect.IO
import sys.process._
object Util{
  def using[C<:Closeable,T](res:C)(f:C=>T):T= {
    try {
      f(res)
    } finally {
      res.close()
    }
  }
  def save(url:URL,dst:File) = using(new BufferedInputStream(url.openStream())){
    in => using(new FileOutputStream(dst)){
      out => Iterator.continually(in.read()).takeWhile(_ != -1).foreach(out.write)
    }
  }
  def ensureDownload(validator:File=>Boolean)(dst:File,url:URL)=Try{
    validator(dst) ? () | save(url,dst)
  }
  def downloadAllAt(dir:String)(validator:File=>Boolean)(targets:Iterator[(String,URL)]) = {
    new File(dir).mkdirs()
    targets.parMap {
        case (filename, url) =>
          val dst = new File(s"$dir/$filename")
          ensureDownload(validator)(dst, url)
    }
  }
  implicit class IteratorOps[T](val itr:Iterator[T]) extends AnyVal{
    def parMap[U](f:T=>U):Iterator[U]=
      itr.grouped(100).map{_.toSeq.par.map(f).toIterator}.flatten
  }

  def throwableToStackTrace(e:Throwable):String =
    new StringWriter() <| (sw=>e.printStackTrace(new PrintWriter(sw))) |> (_.toString)

  def main(args: Array[String]) {
    val targets = Seq(
      "test"-> new URL("http://www.goole.com")
    )
    downloadAllAt("thumbnails")(_.length() > 0)(targets.iterator).foreach(println)
  }
}

object EBookDownloader{
  import Util._
  def main(args: Array[String]): Unit = {
    //downloadThumbnails()
    resizeThumbnails()
  }
  def downloadThumbnails(): Unit ={
    val lines = scala.io.Source.fromFile("ebookdata/bookinfo.csv").getLines()
    val columnIndex = lines.next().split(",").zipWithIndex.toMap
    columnIndex.foreach(println)
    val books = lines.map{line => columnIndex andThen line.split(",")}
    val thumbnails = books.map{
      info => val url = info("サムネイル")+info("表示画像")
        val ext = url.split('.').last
        (info("SKU")+"."+ext)->new URL(url)
    }
    Util.downloadAllAt("thumbnails")(_.length > 0)(thumbnails).zipWithIndex.foreach(println)
  }
  import java.awt.image.BufferedImage

  def resizeThumbnails(): Unit ={
    val thumbnails = new File("thumbnails").listFiles()
    val folder = new File("resized")
    folder.mkdirs()
    thumbnails.iterator.parMap{
      case file =>
        val dst = new File("resized/"+file.getName)
        if(!dst.exists) {
          Thumbnails
            .of(file)
            .size(100, 100)
            .keepAspectRatio(false)
            .scalingMode(ScalingMode.BICUBIC)
            .toFile(dst)
          println("saving resized image:" + dst)
        }else{
          println("skip:"+dst)
        }
    }.foreach(identity)
  }
}