<?php
#$result = exec('E:\xampp\htdocs\Challenge2020\main.py');
$result = shell_exec('python E:\xampp\htdocs\Challenge2020\main.py');
$result_array = json_decode($result,true);
echo ($result_array['result']);
/*foreach($result_array as $row){
    echo $row . "<br>";
}*/
?>