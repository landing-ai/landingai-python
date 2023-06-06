import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from landingai.common import SegmentationPrediction
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

_API_KEY = "v7b0hdyfj6271xy2o9lmiwkkcbdpvt1"
_API_SECRET = "ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq3u1cgq"
_EXPECTED_SEG_PRED = {
    "label_name": "screw",
    "label_index": 1,
    "score": 0.99487104554061,
    "encoded_mask": "2134595Z28N2020Z28N2020Z28N2015Z36N2012Z36N2012Z36N2007Z43N2005Z43N2002Z49N1999Z49N1999Z49N1997Z51N1997Z51N1992Z58N1990Z58N1990Z58N1982Z66N1982Z66N1974Z74N1974Z74N1974Z74N1969Z79N1969Z79N1967Z81N1967Z81N1967Z81N1964Z84N1964Z84N1964Z84N1964Z87N1961Z87N1958Z90N1958Z90N1958Z90N1956Z92N1956Z92N1953Z95N1953Z95N1953Z95N1951Z97N1951Z97N1946Z99N1949Z99N1949Z99N1905Z10N28Z105N1905Z10N28Z105N1895Z31N12Z110N1895Z31N12Z110N1895Z31N12Z110N1892Z156N1892Z156N1892Z156N1890Z158N1890Z158N1890Z158N1890Z158N1890Z158N1890Z156N1892Z156N1889Z156N1892Z156N1892Z156N1892Z154N1894Z154N1894Z151N1897Z151N1897Z151N1897Z146N1902Z146N1902Z146N1902Z144N1904Z144N1904Z141N1907Z141N1907Z141N1907Z138N1910Z138N1910Z133N1915Z133N1915Z133N1915Z128N1920Z128N1920Z120N1928Z120N1928Z120N1928Z118N1930Z118N1933Z110N1938Z110N1938Z110N1938Z107N1941Z107N1941Z107N1941Z102N1946Z102N1948Z98N1950Z98N1950Z98N1950Z92N1956Z92N1959Z82N1966Z82N1966Z82N1968Z75N1973Z75N1976Z69N1979Z69N1979Z69N1979Z67N1981Z67N1984Z64N1984Z64N1984Z64N1984Z61N1987Z61N1987Z61N1987Z61N1987Z61N1989Z59N1989Z59N1989Z59N1992Z53N1995Z53N1997Z51N1997Z51N1997Z51N2000Z48N2000Z48N2005Z43N2005Z43N2005Z43N2007Z41N2007Z41N2007Z41N2010Z38N2010Z38N2013Z35N2013Z35N2013Z35N2018Z28N2020Z28N2030Z15N2033Z15N2033Z15N531789Z18N2030Z18N2030Z18N2020Z33N2015Z33N2010Z43N2005Z43N2005Z43N2000Z53N1995Z53N1995Z53N1992Z61N1987Z61N1984Z67N1981Z67N1981Z67N1976Z75N1973Z75N1966Z82N1966Z82N1966Z82N1955Z95N1953Z95N1948Z105N1943Z105N1943Z105N1938Z113N1935Z113N1930Z123N1925Z123N1925Z123N1920Z131N1917Z131N1917Z131N1912Z138N1910Z138N1905Z143N1905Z143N1905Z143N1902Z149N1899Z149N1894Z154N1894Z154N1894Z154N1891Z159N1889Z159N1887Z164N1884Z164N1884Z164N1879Z171N1877Z171N1877Z171N1872Z179N1869Z179N1861Z187N1861Z187N1861Z187N1856Z194N1854Z194N1849Z202N1846Z202N1846Z202N1841Z207N1841Z207N1838Z210N1838Z210N1838Z210N1833Z218N1830Z218N1825Z223N1825Z223N1825Z223N1822Z226N1822Z226N1822Z226N1817Z231N1817Z231N1815Z235N1813Z235N1813Z235N1805Z243N1805Z243N1797Z251N1797Z251N1797Z251N1792Z256N1792Z256N1784Z264N1784Z264N1784Z264N1779Z269N1779Z269N1777Z271N1777Z271N1777Z271N1772Z276N1772Z276N1772Z276N1769Z279N1769Z279N1764Z284N1764Z284N1764Z284N1754Z292N1756Z292N1738Z310N1738Z310N1738Z310N1725Z320N1728Z320N1718Z330N1718Z330N1718Z330N1713Z332N1716Z332N1716Z332N1713Z333N1715Z333N1710Z338N1710Z338N1710Z338N1707Z338N1710Z338N1672Z23N8Z343N1674Z23N8Z343N1674Z23N8Z343N1669Z376N1672Z376N1667Z381N1667Z381N1667Z381N1659Z387N1661Z387N1653Z392N1656Z392N1656Z392N1654Z392N1656Z392N1656Z392N1653Z392N1656Z392N1656Z389N1659Z389N1659Z389N1659Z387N1661Z387N1661Z384N1664Z384N1664Z384N1664Z382N1666Z382N1666Z377N1671Z377N1671Z377N1671Z371N1677Z371N1677Z366N1682Z366N1682Z366N1682Z364N1684Z364N1684Z364N1684Z359N1689Z359N1687Z358N1690Z358N1690Z358N1690Z356N1692Z356N1692Z353N1695Z353N1695Z353N1695Z348N1700Z348N1697Z346N1702Z346N1702Z346N1152Z20N530Z341N1157Z20N530Z341N1157Z20N530Z341N1154Z31N522Z336N1159Z31N522Z336N1157Z38N517Z333N1160Z38N517Z333N1160Z38N517Z333N1157Z44N514Z328N1162Z44N514Z328N1162Z51N507Z307N1183Z51N507Z307N1183Z51N507Z307N1181Z66N497Z299N1186Z66N497Z299N1186Z71N492Z294N1191Z71N492Z294N1191Z71N492Z294N1188Z79N489Z287N1193Z79N489Z287N1193Z84N489Z277N1198Z84N489Z277N1198Z84N489Z277N1195Z93N486Z269N1200Z93N486Z269N1200Z93N486Z269N1198Z97N489Z261N1201Z97N489Z261N1201Z97N492Z251N1208Z97N492Z251N1208Z97N492Z251N1208Z97N494Z238N1219Z97N494Z238N1219Z97N499Z226N1226Z97N499Z226N1226Z97N499Z226N1226Z97N499Z220N1232Z97N499Z220N1232Z97N502Z212N1237Z97N502Z212N1237Z97N502Z212N1239Z98N501Z205N1244Z98N501Z205N1244Z98N504Z200N1246Z98N504Z200N1246Z98N504Z200N1246Z100N502Z197N1249Z100N502Z197N1249Z100N502Z197N1252Z97N505Z189N1257Z97N505Z189N1262Z95N502Z184N1267Z95N502Z184N1267Z95N502Z184N1270Z94N500Z176N1278Z94N500Z176N1280Z92N500Z174N1282Z92N500Z174N1282Z92N500Z174N1287Z87N500Z169N1292Z87N500Z169N1295Z84N500Z166N1298Z84N500Z166N1298Z84N500Z166N1300Z82N500Z158N1308Z82N500Z158N1308Z82N500Z158N1311Z79N500Z153N1316Z79N500Z153N1316Z79N502Z146N1321Z79N502Z146N1321Z79N502Z146N1318Z82N502Z141N1323Z82N502Z141N1323Z80N504Z141N1323Z80N504Z141N1323Z80N504Z141N1323Z77N507Z138N1326Z77N507Z138N1324Z77N509Z136N1326Z77N509Z136N1326Z77N509Z136N1326Z74N515Z133N1326Z74N515Z133N1326Z74N515Z130N1329Z74N515Z130N1329Z74N515Z130N1329Z72N517Z130N1329Z72N517Z130N1329Z72N517Z130N1326Z75N519Z126N1328Z75N519Z126N1328Z75N522Z123N1328Z75N522Z123N1328Z75N522Z123N1326Z77N524Z121N1326Z77N524Z121N1323Z80N524Z121N1323Z80N524Z121N1323Z80N524Z121N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z85N529Z116N1318Z82N535Z113N1318Z82N535Z113N1318Z82N535Z113N1318Z82N535Z110N1321Z82N535Z110N1321Z82N538Z107N1321Z82N538Z107N1321Z82N538Z107N1321Z79N543Z103N1323Z79N543Z103N1323Z77N548Z97N1326Z77N548Z97N1326Z77N548Z97N1324Z76N553Z95N1324Z76N553Z95N1324Z74N560Z87N1327Z74N560Z87N1327Z74N560Z87N1327Z71N566Z82N1329Z71N566Z82N1329Z71N566Z82N1329Z69N571Z79N1329Z69N571Z79N1329Z69N576Z71N1332Z69N576Z71N1332Z69N576Z71N1334Z67N578Z67N1336Z67N578Z67N1336Z64N584Z61N1339Z64N584Z61N1339Z64N584Z61N1339Z64N586Z54N1344Z64N586Z54N1344Z64N586Z49N1349Z64N586Z49N1349Z64N586Z49N1349Z62N591Z43N1352Z62N591Z43N1352Z62N593Z39N1354Z62N593Z39N1354Z62N593Z39N1354Z62N596Z33N1357Z62N596Z33N1357Z62N596Z33N1360Z56N604Z26N1362Z56N604Z26N1364Z51N625Z3N1369Z51N625Z3N1369Z51N625Z3N1372Z46N2002Z46N2007Z36N2012Z36N2012Z36N2014Z29N2019Z29N2025Z17N2031Z17N2031Z17N2033Z13N2035Z13N2038Z5N2043Z5N2043Z5N519175Z",
    "num_predicted_pixels": 94553,
    "percentage_predicted_pixels": 0.02254319190979004,
    "mask_shape": (2048, 2048),
}
_EXPECTED_VP_PREDS = [
    {
        "label_name": "Green Field",
        "label_index": 3,
        "score": 0.8484444637722576,
        "encoded_mask": "169Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N104Z20N43Z30N6Z16N781Z10N359Z22N48Z164N112Z8N49Z26N10Z16N779Z10N359Z22N48Z164N112Z8N49Z26N10Z16N779Z10N359Z22N46Z164N175Z22N10Z16N779Z8N361Z22N46Z164N175Z22N10Z16N779Z8N361Z22N46Z164N177Z18N14Z14N779Z8N361Z22N46Z164N177Z18N14Z14N779Z8N361Z22N44Z162N183Z14N18Z14N777Z6N363Z22N44Z162N183Z14N18Z14N777Z6N363Z22N42Z90N32Z40N187Z10N20Z16N775Z4N365Z22N42Z90N32Z40N187Z10N20Z16N775Z4N365Z22N40Z88N48Z24N193Z4N26Z14N1144Z22N40Z88N48Z24N193Z4N26Z14N1144Z22N40Z84N60Z6N233Z16N1142Z22N40Z84N60Z6N233Z16N1142Z22N38Z82N303Z18N1140Z22N38Z82N303Z18N1138Z26N36Z76N309Z28N1128Z26N36Z76N309Z28N1128Z28N34Z74N313Z26N1128Z30N32Z72N315Z26N328Z2N798Z30N32Z72N315Z26N328Z2N798Z32N30Z68N321Z24N326Z16N8Z6N772Z32N30Z68N321Z24N326Z16N8Z6N772Z36N26Z18N2Z48N321Z24N324Z36N768Z36N26Z18N2Z48N321Z24N324Z36N768Z38N24Z14N8Z44N325Z22N324Z38N766Z38N24Z14N8Z44N325Z22N324Z38N766Z40N22Z10N14Z42N327Z20N322Z44N762Z40N22Z10N14Z42N327Z20N322Z44N760Z44N20Z10N16Z40N327Z18N324Z46N758Z44N20Z10N16Z40N327Z18N324Z46N760Z44N16Z10N18Z40N329Z16N324Z48N758Z44N16Z10N18Z40N329Z16N324Z48N782Z22N12Z12N20Z36N333Z12N326Z50N780Z22N12Z12N20Z36N333Z12N326Z50N790Z12N10Z14N20Z36N339Z2N330Z52N788Z12N10Z14N20Z36N339Z2N330Z52N792Z10N6Z16N22Z34N671Z54N790Z10N6Z16N22Z34N671Z54N792Z30N22Z34N671Z56N790Z30N22Z34N671Z56N794Z26N22Z32N675Z56N792Z26N22Z32N675Z56N794Z24N24Z28N677Z56N796Z22N24Z26N679Z58N794Z22N24Z26N679Z58N796Z20N26Z22N681Z58N796Z20N26Z22N681Z58N800Z18N24Z22N683Z56N800Z18N24Z22N683Z56N804Z14N24Z22N683Z56N804Z14N24Z22N683Z56N808Z10N24Z20N687Z54N808Z10N24Z20N687Z54N812Z4N28Z18N687Z56N810Z4N28Z18N687Z56N846Z14N689Z54N846Z14N689Z54N852Z4N695Z52N852Z4N695Z52N1553Z50N1553Z50N1555Z48N1555Z48N1557Z44N1559Z44N1561Z42N1561Z42N860Z6N699Z34N862Z10N699Z28N866Z10N699Z28N866Z10N705Z16N872Z10N705Z16N870Z12N1591Z12N1591Z14N1589Z14N1589Z14N1589Z14N1595Z8N1595Z8N8021Z4N1599Z4N1599Z8N1595Z8N1595Z12N313Z4N1274Z12N313Z4N1274Z14N309Z6N1274Z14N309Z6N1276Z16N303Z8N1248Z4N26Z16N72Z16N209Z10N1250Z4N26Z16N72Z16N209Z10N1250Z6N28Z10N62Z36N197Z14N1250Z6N28Z10N62Z36N197Z14N1248Z12N90Z48N66Z6N117Z16N1248Z12N90Z48N66Z6N117Z16N1248Z18N80Z58N54Z16N111Z16N1250Z18N80Z58N54Z16N111Z16N1250Z20N76Z72N38Z22N107Z18N1250Z20N76Z72N38Z22N107Z18N1250Z22N72Z136N103Z20N1250Z22N72Z136N103Z20N1250Z22N72Z136N101Z22N1250Z22N72Z136N101Z22N1250Z24N70Z136N105Z16N1252Z24N70Z136N105Z16N1254Z24N68Z74N24Z36N1377Z24N68Z74N24Z36N1379Z24N64Z64N42Z24N1385Z24N64Z64N42Z24N1385Z26N62Z58N1457Z26N62Z58N1459Z26N60Z52N1465Z26N60Z52N1467Z24N60Z46N1475Z24N58Z44N1477Z24N58Z44N1479Z24N54Z42N1483Z24N54Z42N1485Z22N54Z40N1487Z22N54Z40N1489Z22N54Z38N1489Z22N54Z38N1493Z22N52Z34N1495Z22N52Z34N1497Z26N50Z28N1499Z26N50Z28N1501Z38N38Z20N1507Z38N38Z20N1511Z36N40Z10N1517Z36N40Z10N1519Z36N1567Z36N1569Z36N1567Z36N1569Z36N1567Z36N1567Z38N1565Z38N1565Z40N26Z26N1513Z40N18Z36N1509Z40N18Z36N1509Z38N6Z54N1505Z38N6Z54N1507Z34N10Z56N1503Z34N10Z56N1503Z32N14Z58N1499Z32N14Z58N1499Z30N18Z58N1497Z30N18Z58N1499Z26N22Z56N1499Z26N22Z56N1499Z26N26Z50N1501Z26N26Z50N1501Z26N30Z44N912Z2N589Z26N30Z44N912Z2N591Z22N36Z34N918Z4N589Z22N36Z34N918Z4N589Z22N40Z22N928Z4N587Z22N40Z22N928Z4N587Z22N42Z8N940Z6N585Z22N42Z8N940Z6N585Z22N990Z8N126Z4N453Z22N990Z8N126Z4N453Z22N992Z6N126Z12N447Z20N992Z8N124Z14N445Z20N992Z8N124Z14N445Z18N994Z10N122Z16N443Z18N994Z10N122Z16N447Z14N996Z10N122Z14N447Z14N996Z10N122Z14N451Z10N996Z12N120Z16N449Z10N996Z12N120Z16N451Z8N996Z16N118Z16N449Z8N996Z16N118Z16N453Z2N1000Z18N114Z20N449Z2N1000Z18N114Z20N1451Z20N112Z24N1447Z20N112Z24N1447Z22N112Z24N1445Z22N112Z24N1447Z20N112Z26N1445Z20N112Z26N1447Z18N110Z28N1447Z18N110Z28N1449Z12N112Z32N1447Z12N112Z32N1451Z2N116Z34N1451Z2N116Z34N1565Z38N1363Z2N196Z42N1363Z2N196Z42N1361Z4N194Z46N1359Z4N194Z46N1355Z8N192Z48N1355Z8N192Z48N1351Z12N192Z36N2Z10N1351Z12N192Z36N2Z10N1351Z12N192Z32N8Z6N1353Z12N192Z32N8Z6N1351Z14N194Z28N10Z6N1351Z14N194Z28N10Z6N1351Z12N196Z26N1369Z12N196Z26N139Z4N1224Z14N198Z20N143Z4N1224Z14N198Z20N134Z13N1224Z14N198Z18N136Z13N1224Z14N198Z18N136Z13N1222Z16N200Z12N130Z23N1222Z16N200Z12N130Z23N1222Z14N204Z8N130Z25N1222Z14N204Z8N130Z25N1222Z14N344Z23N1222Z14N344Z23N1222Z14N354Z13N1222Z14N354Z11N1224Z14N354Z11N1224Z16N352Z11N1224Z16N352Z11N1226Z14N352Z11N1226Z14N352Z9N1228Z14N352Z9N1228Z14N352Z9N1228Z14N352Z9N1228Z14N1589Z14N1589Z14N1591Z12N1591Z12N1593Z10N1593Z10N14771Z8N1595Z8N1593Z10N1593Z10N1593Z10N1591Z12N1591Z12N1591Z12N1591Z12N1593Z10N1593Z10N20651Z6N1597Z6N1595Z8N1595Z8N1593Z12N1591Z12N1591Z12N1589Z16N1587Z16N1587Z16N1587Z16N1589Z14N1589Z14N1591Z12N1591Z12N1593Z8N1595Z8N988Z2N1601Z2N1601Z2N1601Z2N1601Z2N1601Z2N128737Z2N26Z6N1569Z2N26Z6N1569Z2N26Z10N1593Z12N1591Z12N1593Z14N1589Z14N1589Z18N1585Z18N1587Z20N1583Z20N1585Z24N1579Z24N1579Z30N1573Z30N1575Z32N1571Z32N1571Z34N1569Z34N1567Z38N1565Z38N1565Z40N1563Z40N1557Z50N1553Z50N1551Z80N1523Z80N1527Z84N1521Z82N1521Z82N1523Z82N1521Z82N1523Z80N1523Z80N1525Z78N1525Z78N1527Z78N1525Z78N1527Z76N1527Z76N1531Z72N1531Z72N1533Z70N1533Z70N1539Z64N1539Z64N1543Z60N1543Z60N1547Z56N1547Z56N1589Z12N1591Z12N11424Z4N1599Z4N1599Z6N1597Z6N1595Z8N1595Z8N1597Z4N1599Z4N41672Z6N1597Z6N1595Z8N1595Z8N1595Z10N1593Z10N1591Z12N1591Z12N1591Z14N1589Z14N1589Z14N1589Z14N1587Z16N1587Z18N1585Z18N1585Z18N1585Z18N1585Z18N1585Z18N1585Z18N1585Z18N1587Z16N1587Z16N1589Z12N1591Z12N1595Z4N1599Z4N313454Z6N1597Z6N1595Z12N1591Z12N1591Z12N1591Z12N1589Z16N1587Z16N1587Z16N1587Z16N1587Z16N1587Z16N1587Z16N1587Z16N1589Z14N1589Z14N1591Z12N1591Z12N1599Z2N187533Z6N1597Z6N1597Z6N1597Z6N1595Z10N1593Z10N1593Z10N1593Z10N1593Z10N1593Z12N1591Z12N1589Z14N1589Z14N1589Z14N1589Z14N1589Z16N1587Z16N1587Z16N1587Z16N1589Z16N1587Z16N1587Z18N1585Z18N1585Z20N1583Z20N1583Z20N1583Z20N1583Z22N1581Z22N1583Z22N1581Z22N1581Z22N1581Z22N1581Z24N1579Z26N1577Z26N1577Z26N1577Z26N1579Z26N1577Z26N1577Z30N1573Z30N1573Z32N1571Z32N1571Z34N1569Z34N1571Z34N1569Z34N1571Z32N1571Z32N1573Z30N1573Z30N1575Z30N1573Z30N1573Z30N1573Z30N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1577Z26N1577Z26N1579Z24N1579Z24N1579Z22N1581Z22N1587Z14N1589Z14N112643Z6N1597Z6N1595Z12N1591Z12N1591Z14N1589Z14N1589Z16N1587Z16N1585Z20N1583Z20N1583Z22N1581Z22N1581Z24N1579Z24N1579Z26N1577Z26N1577Z28N1575Z28N1577Z28N1575Z28N1579Z22N1581Z22N1583Z18N1589Z12N1591Z12N1593Z6N1597Z6N162107Z26N1577Z26N1557Z46N1557Z46N1553Z50N1553Z50N1551Z52N1551Z52N1551Z52N1551Z52N1551Z52N1551Z52N1553Z50N1553Z50N1555Z48N1555Z48N1557Z46N1557Z46N1559Z44N1559Z44N1563Z40N1563Z40N1565Z38N1569Z34N1569Z34N1569Z34N1569Z34N1571Z32N1571Z32N1571Z32N1571Z32N1571Z32N1571Z32N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1577Z26N1577Z26N1577Z26N1577Z26N1577Z26N1577Z26N1579Z24N1579Z24N1579Z24N1579Z24N1581Z22N1581Z22N1581Z22N1581Z22N1583Z20N1583Z20N1583Z20N1583Z20N1585Z18N1585Z18N1587Z16N1587Z16N1587Z16N1589Z14N1589Z14N1591Z12N1591Z12N1595Z8N1595Z8N34229Z30N66Z6N1501Z30N66Z6N1497Z40N54Z10N1499Z40N54Z10N1497Z50N44Z8N1501Z50N44Z8N1497Z58N38Z10N1497Z58N38Z10N1493Z66N30Z12N1495Z66N30Z12N1489Z74N26Z12N1491Z74N26Z12N1489Z78N22Z14N1489Z78N22Z14N1485Z62N2Z16N24Z12N1487Z62N2Z16N24Z12N1144Z24N317Z60N46Z12N1144Z24N317Z60N46Z12N1140Z34N309Z60N50Z10N1140Z34N309Z60N50Z10N1138Z36N20Z2N287Z60N50Z10N1138Z36N20Z2N287Z60N50Z10N1138Z34N22Z2N284Z65N46Z14N22Z6N1108Z32N24Z4N282Z67N44Z16N14Z20N1100Z32N24Z4N282Z67N44Z16N14Z20N1100Z26N30Z4N280Z73N40Z20N4Z32N1094Z26N30Z4N280Z73N40Z20N4Z32N1094Z24N32Z4N280Z75N36Z60N1092Z24N32Z4N280Z75N36Z60N1094Z20N34Z6N278Z77N34Z62N1092Z20N34Z6N278Z77N34Z62N1092Z18N36Z10N272Z81N32Z64N1090Z18N36Z10N272Z81N32Z64N1090Z18N36Z18N264Z81N32Z66N1088Z18N36Z18N264Z81N32Z66N1088Z16N38Z20N260Z83N32Z66N1088Z16N38Z20N260Z83N32Z66N1088Z14N38Z24N258Z83N32Z66N1088Z14N38Z24N258Z83N32Z66N1088Z14N38Z26N256Z83N32Z66N1088Z14N38Z26N256Z83N32Z66N1090Z10N40Z26N256Z81N34Z66N1090Z10N40Z26N256Z81N34Z66N1092Z6N42Z28N254Z81N36Z64N1092Z6N42Z28N254Z81N36Z64N1140Z28N254Z81N36Z64N1140Z28N254Z81N36Z64N1140Z30N252Z81N36Z64N1140Z30N252Z43N2Z34N38Z64N1140Z30N252Z43N2Z34N38Z64N1140Z32N250Z41N4Z34N40Z60N1142Z32N250Z41N4Z34N40Z60N1144Z30N248Z45N2Z34N40Z58N1146Z30N248Z45N2Z34N40Z58N1148Z26N250Z81N42Z52N60Z4N1088Z26N250Z81N42Z52N60Z4N1090Z24N250Z81N44Z46N64Z4N1090Z24N250Z81N44Z46N64Z4N1092Z22N248Z83N48Z34N72Z4N1092Z22N248Z83N48Z34N72Z4N1094Z20N246Z85N52Z24N76Z6N1094Z20N246Z85N52Z24N76Z6N1096Z18N246Z85N56Z10N86Z6N156Z2N938Z18N246Z85N56Z10N86Z6N156Z2N938Z18N244Z87N152Z8N152Z6N936Z18N244Z87N152Z8N152Z6N938Z14N246Z33N10Z42N156Z4N154Z6N938Z14N246Z33N10Z42N156Z4N154Z6N938Z14N244Z35N8Z44N316Z4N938Z14N244Z35N8Z44N316Z4N938Z12N246Z37N2Z48N316Z2N940Z12N246Z37N2Z48N316Z2N940Z12N244Z87N1262Z8N246Z85N1264Z8N246Z85N1264Z8N244Z85N1266Z8N244Z85N1266Z6N244Z85N1268Z6N244Z85N1270Z4N244Z83N1272Z4N244Z83N1272Z4N240Z77N1282Z4N240Z77N1282Z2N238Z71N1292Z2N238Z71N1526Z73N1530Z73N1528Z73N1530Z73N1530Z71N1532Z71N1532Z69N1534Z69N1536Z65N214Z4N1320Z65N214Z4N1320Z65N212Z8N1318Z65N212Z8N1320Z65N84Z4N122Z8N1324Z6N39Z16N82Z8N120Z8N1324Z6N39Z16N82Z8N120Z8N1463Z14N118Z6N1465Z14N118Z6N1459Z22N118Z4N1459Z22N118Z4N1022Z4N429Z24N120Z4N1022Z4N429Z24N120Z4N1024Z4N423Z28N120Z2N1026Z4N423Z28N120Z2N1453Z26N1577Z26N1575Z26N1577Z26N1575Z24N1579Z24N1573Z20N1583Z20N7622Z6N1597Z6N1601Z2N18094Z4N1599Z4N1599Z4N1599Z4N1597Z8N1595Z8N1593Z10N1593Z10N1593Z12N1591Z12N1589Z14N1589Z14N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1593Z10N1593Z10N1593Z10N1593Z10N1595Z8N1595Z8N17657Z4N236Z4N1359Z4N236Z4N1357Z6N228Z18N1351Z6N228Z18N1353Z6N224Z20N1589Z10N1593Z10N277932Z",
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 39711,
        "percentage_predicted_pixels": 0.016096767877967603,
    },
    {
        "label_name": "Brown Field",
        "label_index": 4,
        "score": 0.9469594520537422,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 657373,
        "percentage_predicted_pixels": 0.26646472237524105,
    },
    {
        "label_name": "Trees",
        "label_index": 5,
        "score": 0.9759463515311614,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 990878,
        "percentage_predicted_pixels": 0.401650252106086,
    },
    {
        "label_name": "Structure",
        "label_index": 6,
        "score": 0.9677068448643612,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 765303,
        "percentage_predicted_pixels": 0.3102139142129949,
    },
]


def test_od_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Endpoint: https://app.landing.ai/app/376/pr/11165/deployment?device=tiger-team-integration-tests
    endpoint_id = "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4"
    predictor = Predictor(endpoint_id, _API_KEY, _API_SECRET)
    img = np.asarray(Image.open("tests/data/images/cereal1.jpeg"))
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    preds = predictor.predict(img)
    assert len(preds) == 3, "Result should not be empty or None"
    expected_scores = [0.9997884631156921, 0.9979170560836792, 0.9976948499679565]
    expected_bboxes = [
        (432, 1035, 651, 1203),
        (1519, 1414, 1993, 1800),
        (948, 1592, 1121, 1797),
    ]
    for i, pred in enumerate(preds):
        assert pred.label_name == "Screw"
        assert pred.label_index == 1
        assert pred.score == expected_scores[i]
        assert pred.bboxes == expected_bboxes[i]
    logging.info(preds)
    img_with_preds = overlay_predictions(predictions=preds, image=img)
    img_with_preds.save("tests/output/test_od.jpg")


def test_seg_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26113016987660/deployment?device=tiger-team-integration-tests
    endpoint_id = "72fdc6c2-20f1-4f5e-8df4-62387acec6e4"
    predictor = Predictor(endpoint_id, _API_KEY, _API_SECRET)
    img = Image.open("tests/data/images/cereal1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    _assert_seg_mask(preds[0], _EXPECTED_SEG_PRED)
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_seg.jpg")


def test_vp_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26098103179275/deployment?device=tiger-example
    endpoint_id = "63035608-9d24-4342-8042-e4b08e084fde"
    predictor = Predictor(endpoint_id, _API_KEY, _API_SECRET)
    img = np.asarray(Image.open("tests/data/images/farm-coverage.jpg"))
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 4, "Result should not be empty or None"
    for actual, expected in zip(preds, _EXPECTED_VP_PREDS):
        _assert_seg_mask(actual, expected)
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_vp.jpg")


def test_class_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26119078438913/deployment?device=tiger-team-integration-tests
    endpoint_id = "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43"
    predictor = Predictor(endpoint_id, _API_KEY, _API_SECRET)
    img = Image.open("tests/data/images/wildfire1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    assert preds[0].label_name == "HasFire"
    assert preds[0].label_index == 0
    assert preds[0].score == 0.9956502318382263
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_class.jpg")


def _assert_seg_mask(pred: SegmentationPrediction, expected: dict[str, Any]):
    assert pred.label_name == expected["label_name"]
    assert pred.label_index == expected["label_index"]
    assert pred.score == expected["score"]
    assert pred.num_predicted_pixels == expected["num_predicted_pixels"]
    assert pred.percentage_predicted_pixels == expected["percentage_predicted_pixels"]
    assert pred.decoded_boolean_mask.shape == expected["mask_shape"]
    assert np.unique(pred.decoded_boolean_mask).tolist() == [0, 1]
    assert np.unique(pred.decoded_index_mask).tolist() == [0, pred.label_index]
    if "encoded_mask" in expected:
        assert pred.encoded_mask == expected["encoded_mask"]