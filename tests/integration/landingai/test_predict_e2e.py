import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops
from landingai.common import InferenceMetadata

from landingai.predict import OcrPredictor, Predictor
from landingai.visualize import overlay_predictions

_API_KEY = "land_sk_aMemWbpd41yXnQ0tXvZMh59ISgRuKNRKjJEIUHnkiH32NBJAwf"

_EXPECTED_VP_PREDS = [
    {
        "label_name": "Green Field",
        "label_index": 3,
        "score": 0.8327239290916779,
        "encoded_mask": "169Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N100Z28N37Z30N24Z2N779Z14N359Z18N54Z158N104Z22N43Z26N24Z4N779Z10N361Z18N54Z158N104Z22N43Z26N24Z4N779Z10N359Z20N52Z158N175Z20N24Z6N779Z10N359Z20N52Z158N175Z20N24Z6N779Z10N359Z20N50Z156N183Z14N26Z8N777Z8N361Z20N50Z156N183Z14N26Z8N777Z8N361Z20N50Z92N4Z56N191Z10N26Z10N777Z4N363Z20N50Z92N4Z56N191Z10N26Z10N777Z4N363Z20N46Z90N32Z30N197Z6N28Z10N1144Z20N46Z90N32Z30N197Z6N28Z10N1144Z20N44Z88N48Z14N235Z12N1142Z20N44Z88N48Z14N235Z12N1142Z20N44Z86N299Z12N1142Z20N44Z86N299Z12N1140Z22N42Z86N301Z14N1138Z22N42Z86N301Z14N1138Z24N40Z84N303Z26N1126Z24N40Z84N303Z26N1126Z26N38Z82N305Z26N1124Z30N36Z74N315Z26N1122Z30N36Z74N315Z26N1122Z30N34Z70N321Z26N1122Z30N34Z70N321Z26N1122Z32N32Z18N6Z46N323Z22N1124Z32N32Z18N6Z46N323Z22N1124Z34N30Z14N12Z42N327Z20N334Z20N770Z34N30Z14N12Z42N327Z20N334Z20N770Z38N26Z14N14Z40N327Z20N332Z26N766Z38N26Z14N14Z40N327Z20N332Z26N766Z40N24Z12N16Z40N329Z18N330Z32N762Z40N24Z12N16Z40N329Z18N330Z32N762Z44N18Z14N18Z38N331Z14N332Z36N383Z4N371Z44N18Z14N18Z38N331Z14N332Z36N383Z4N371Z46N16Z12N20Z38N333Z10N334Z40N379Z2N373Z46N16Z12N20Z38N333Z10N334Z40N379Z2N375Z46N14Z12N20Z38N337Z4N336Z44N373Z4N375Z46N14Z12N20Z38N337Z4N336Z44N373Z4N377Z12N24Z8N14Z12N22Z34N679Z46N371Z4N377Z12N24Z8N14Z12N22Z34N679Z46N371Z4N433Z14N22Z34N679Z48N369Z2N435Z14N22Z34N679Z48N369Z2N435Z14N22Z34N679Z48N806Z14N22Z34N679Z48N806Z14N24Z32N679Z50N804Z14N24Z32N679Z50N804Z14N24Z32N679Z50N806Z12N24Z30N681Z52N804Z12N24Z30N681Z52N806Z10N24Z30N681Z52N806Z10N24Z30N681Z52N808Z8N24Z30N681Z52N808Z8N24Z30N681Z52N810Z8N20Z30N685Z50N810Z8N20Z30N685Z50N836Z32N685Z52N834Z32N685Z52N832Z32N689Z50N832Z32N689Z50N830Z34N691Z48N830Z34N691Z48N826Z36N693Z48N826Z36N693Z48N826Z38N501Z2N190Z46N826Z38N501Z2N190Z46N826Z38N695Z44N826Z38N695Z44N826Z40N695Z40N828Z40N695Z40N828Z40N699Z34N832Z12N6Z22N699Z30N834Z12N6Z22N699Z30N834Z12N16Z12N705Z16N842Z12N16Z12N705Z16N842Z12N18Z10N1563Z12N18Z10N1563Z12N20Z8N1563Z12N20Z8N1565Z10N26Z2N1565Z10N26Z2N1567Z10N1593Z10N1595Z12N1591Z12N1591Z14N24Z2N287Z8N1268Z14N24Z2N287Z8N1268Z16N22Z6N279Z14N28Z2N1236Z16N22Z6N279Z14N28Z2N1238Z14N24Z8N273Z18N24Z6N1236Z14N24Z8N273Z18N24Z6N1236Z14N24Z10N271Z18N20Z12N1234Z14N24Z10N271Z18N20Z12N1234Z14N24Z12N271Z16N16Z16N1234Z14N24Z12N271Z16N16Z16N1234Z16N22Z16N269Z16N12Z18N1236Z16N22Z16N74Z10N185Z16N6Z20N1238Z16N22Z16N74Z10N185Z16N6Z20N1238Z16N24Z16N58Z38N173Z38N1240Z16N24Z16N58Z38N173Z38N1240Z18N28Z8N56Z48N169Z36N1240Z18N28Z8N56Z48N169Z36N1240Z26N80Z58N165Z34N1240Z26N80Z58N165Z34N1240Z30N74Z74N153Z30N1242Z30N74Z74N153Z30N1244Z30N70Z88N145Z26N1244Z30N70Z88N145Z26N1244Z30N70Z126N109Z22N1246Z30N70Z126N109Z22N1246Z32N68Z92N4Z32N123Z4N1248Z32N68Z92N4Z32N123Z4N1250Z32N66Z74N30Z20N1381Z32N66Z74N30Z20N1383Z32N64Z60N50Z2N1395Z32N64Z60N50Z2N1397Z34N58Z56N1455Z34N58Z56N1457Z34N56Z50N1463Z34N56Z50N1465Z34N54Z44N1473Z34N50Z44N1475Z34N50Z44N1477Z32N50Z42N1479Z32N50Z42N1481Z32N48Z40N1483Z32N48Z40N1485Z52N26Z38N1487Z52N26Z38N1491Z52N24Z34N1493Z52N24Z34N1495Z52N24Z30N1497Z52N24Z30N1499Z52N24Z24N1503Z52N24Z24N1505Z52N24Z16N1511Z52N24Z16N1515Z48N1555Z48N1557Z48N1555Z48N1557Z52N1551Z52N1551Z82N1521Z82N1523Z94N1511Z98N1505Z98N1507Z100N1503Z100N1503Z38N4Z62N1499Z38N4Z62N1501Z32N12Z60N1499Z32N12Z60N1501Z26N18Z62N1497Z26N18Z62N1497Z24N24Z58N1497Z24N24Z58N1499Z20N30Z52N1501Z20N30Z52N1501Z20N36Z42N1505Z20N36Z42N1505Z20N40Z32N1511Z20N40Z32N1513Z16N46Z22N1519Z16N46Z22N1519Z16N996Z2N589Z16N996Z2N597Z8N110Z10N876Z4N128Z8N459Z8N110Z10N876Z4N128Z8N465Z2N110Z14N872Z6N126Z14N571Z10N876Z8N124Z16N569Z10N876Z8N124Z16N1457Z8N122Z16N1457Z8N122Z16N1457Z10N120Z18N1455Z10N120Z18N1457Z10N120Z18N1455Z10N120Z18N1455Z14N116Z20N1453Z14N116Z20N1453Z18N112Z22N1451Z18N112Z22N537Z14N902Z18N112Z24N533Z14N902Z18N112Z24N527Z20N902Z20N110Z26N525Z20N902Z20N110Z26N525Z20N904Z20N108Z28N523Z20N904Z20N108Z28N533Z8N906Z20N106Z30N533Z8N906Z20N106Z30N1449Z16N108Z32N1447Z16N108Z32N1449Z8N112Z36N1447Z8N112Z36N1563Z40N1361Z2N196Z44N1361Z2N196Z44N1357Z6N196Z46N1355Z6N196Z46N1353Z8N194Z48N1353Z8N194Z48N1349Z12N194Z48N1349Z12N194Z48N1349Z12N194Z48N1349Z12N194Z48N1349Z12N194Z28N10Z8N1351Z12N194Z28N10Z8N1349Z14N194Z24N18Z2N1351Z14N194Z24N18Z2N1351Z14N196Z18N1375Z14N196Z18N1373Z14N198Z14N1377Z14N198Z14N149Z2N1226Z14N200Z8N153Z2N1226Z14N200Z8N144Z11N1224Z16N200Z6N146Z11N1224Z16N200Z6N146Z11N1224Z16N352Z11N1224Z16N352Z9N1228Z14N352Z9N1228Z14N352Z9N1228Z14N1589Z14N1589Z14N1589Z14N1589Z14N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1593Z10N1593Z10N25992Z8N1595Z8N1593Z10N1593Z10N1591Z12N1591Z12N1591Z12N1591Z12N1589Z14N1589Z14N1589Z14N1589Z14N1589Z14N1589Z14N19050Z2N1601Z2N1597Z8N1595Z8N1595Z8N1595Z8N1599Z2N1601Z2N4198Z2N1601Z2N1601Z2N1601Z2N122339Z4N1599Z4N1597Z12N1591Z12N1597Z14N1589Z14N1591Z16N1587Z16N1587Z18N1587Z18N1585Z18N1585Z20N1583Z20N1585Z22N1581Z22N1581Z28N1575Z28N1577Z30N1573Z30N1575Z34N1569Z34N1573Z36N1567Z36N1565Z40N1563Z40N1559Z48N1555Z48N1549Z56N1547Z56N1549Z56N1547Z56N1553Z58N22Z4N1519Z58N22Z4N1523Z62N6Z16N1521Z84N1519Z84N1521Z84N1519Z84N1521Z82N1521Z82N1521Z82N1521Z82N1523Z82N1521Z82N1523Z80N1523Z80N1525Z50N4Z24N1525Z50N4Z24N1527Z48N6Z22N1527Z48N6Z22N1531Z40N10Z22N1531Z40N10Z22N1535Z34N14Z20N1535Z34N14Z20N1539Z24N24Z14N1541Z24N24Z14N1547Z2N44Z6N1551Z2N44Z6N85162Z2N1601Z2N1597Z6N1597Z6N1593Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1591Z12N1595Z6N1597Z6N313458Z8N1595Z8N1593Z12N1591Z12N1589Z14N1589Z14N1587Z16N1587Z16N1589Z14N1589Z14N1589Z14N1589Z14N1593Z8N1595Z8N195612Z4N1599Z4N1599Z4N1599Z4N1599Z6N1535Z2N60Z6N1535Z2N60Z6N1533Z4N60Z8N1531Z4N60Z8N1531Z4N60Z8N1531Z4N60Z8N1531Z4N60Z6N1533Z4N60Z6N1533Z6N1597Z6N1597Z6N1597Z6N1597Z8N1595Z8N1595Z10N1593Z10N1593Z14N1589Z14N1589Z18N1585Z18N1585Z20N1583Z20N1583Z20N1583Z20N1585Z20N1583Z20N1583Z20N1585Z18N1585Z18N1587Z16N1587Z16N1589Z16N1587Z16N1589Z14N1589Z14N1589Z14N1589Z14N1591Z14N1589Z14N1589Z16N1587Z16N1589Z16N1587Z16N1589Z18N1585Z18N1585Z20N1583Z20N1585Z18N1585Z18N1585Z18N1585Z18N1589Z16N1589Z14N1589Z14N1595Z6N1597Z6N114246Z4N1597Z10N204Z2N1387Z10N204Z2N888Z2N497Z12N202Z2N888Z2N497Z12N202Z2N886Z4N497Z14N200Z2N886Z4N497Z14N200Z2N886Z6N495Z16N196Z4N886Z6N495Z16N196Z4N886Z6N495Z16N196Z6N884Z6N495Z16N196Z6N882Z8N495Z18N194Z6N882Z8N495Z18N194Z6N882Z8N495Z18N194Z6N882Z8N495Z18N194Z6N882Z8N497Z18N192Z6N882Z8N497Z18N192Z6N882Z8N497Z20N190Z8N880Z8N497Z20N190Z8N882Z6N499Z18N190Z4N886Z6N499Z18N190Z4N1397Z12N1591Z12N1593Z8N1595Z8N166926Z14N1589Z14N1565Z38N1565Z38N1551Z52N1551Z52N1549Z54N1549Z54N1549Z54N1549Z54N1551Z52N1551Z52N1551Z52N1551Z52N1553Z50N1553Z50N1555Z48N1555Z48N1557Z46N1557Z46N1559Z44N1559Z44N1561Z42N1561Z42N1563Z40N1565Z38N1565Z38N1567Z36N1567Z36N1569Z34N1569Z34N1569Z34N1569Z34N1571Z32N1571Z32N1571Z32N1571Z32N1571Z32N1571Z32N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1573Z30N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1575Z28N1577Z26N1577Z26N1577Z26N1577Z26N1579Z24N1579Z24N1581Z22N1581Z22N1581Z22N1581Z22N1583Z20N1583Z20N1583Z20N1583Z20N1583Z20N1583Z20N1585Z18N1585Z18N1585Z18N1585Z18N1587Z16N1587Z16N1587Z16N1591Z12N1591Z12N1593Z10N1593Z10N31021Z24N1579Z24N1577Z34N1569Z34N1567Z42N1561Z42N1559Z50N42Z6N1505Z50N42Z6N1503Z56N34Z10N1503Z56N34Z10N1503Z60N28Z12N1503Z60N28Z12N1499Z66N24Z14N1499Z66N24Z14N1497Z48N10Z12N20Z16N1497Z48N10Z12N20Z16N1493Z46N48Z14N1495Z46N48Z14N1148Z18N327Z42N54Z14N1148Z18N327Z42N54Z14N1142Z28N319Z44N58Z12N1142Z28N319Z44N58Z12N1140Z30N317Z44N60Z12N1140Z30N317Z44N60Z12N1142Z26N317Z48N58Z14N1140Z22N318Z51N58Z14N1140Z22N318Z51N58Z14N1142Z16N322Z53N56Z16N1140Z16N322Z53N56Z16N1142Z10N324Z57N52Z18N1142Z10N324Z57N52Z18N1144Z6N324Z65N46Z20N1142Z6N324Z65N46Z20N1470Z73N40Z22N1468Z73N40Z22N1466Z79N36Z22N1466Z79N36Z22N996Z20N174Z18N256Z83N34Z24N994Z20N174Z18N256Z83N34Z24N996Z2N190Z20N254Z81N36Z26N994Z2N190Z20N254Z81N36Z26N1184Z24N252Z81N38Z28N1180Z24N252Z81N38Z28N1180Z26N250Z81N38Z34N1174Z26N250Z81N38Z34N966Z10N198Z26N248Z83N38Z40N960Z10N198Z26N248Z83N38Z40N960Z12N194Z30N246Z83N38Z56N944Z12N194Z30N246Z83N38Z56N942Z14N194Z32N244Z83N38Z56N938Z18N194Z32N242Z85N38Z58N936Z18N194Z32N242Z85N38Z58N936Z18N194Z32N242Z85N38Z56N938Z18N194Z32N242Z85N38Z56N938Z16N196Z32N242Z85N40Z50N942Z16N196Z32N242Z85N40Z50N944Z12N200Z30N240Z87N40Z44N950Z12N200Z30N240Z87N40Z44N952Z8N202Z30N240Z85N44Z36N958Z8N202Z30N240Z85N44Z36N960Z4N206Z28N240Z85N44Z22N974Z4N206Z28N240Z85N44Z22N1186Z24N242Z83N48Z8N1198Z24N242Z83N48Z8N1198Z24N240Z85N1254Z24N240Z85N1256Z22N240Z85N1256Z22N240Z85N1258Z20N240Z83N322Z2N936Z20N240Z83N322Z2N936Z18N242Z83N1260Z18N242Z83N1262Z16N240Z83N1264Z16N240Z83N1264Z14N242Z83N1266Z12N240Z83N1268Z12N240Z83N1268Z10N242Z81N1270Z10N242Z81N1272Z8N242Z81N1272Z8N242Z81N1272Z6N242Z39N8Z32N1276Z6N242Z39N8Z32N1276Z6N238Z41N12Z18N1288Z6N238Z41N12Z18N1290Z2N238Z47N12Z6N1298Z2N238Z47N12Z6N1035Z11N488Z53N1051Z11N488Z53N1051Z9N490Z55N218Z8N823Z9N490Z55N218Z8N1322Z55N214Z12N1322Z55N214Z12N1324Z55N210Z16N1322Z55N210Z16N1322Z55N208Z20N1320Z55N208Z20N1066Z4N252Z53N92Z4N110Z24N1064Z4N252Z53N92Z4N110Z24N1064Z4N254Z2N35Z16N88Z8N108Z24N1461Z10N108Z22N1463Z10N108Z22N1459Z14N108Z22N1459Z14N108Z22N1455Z16N110Z22N1455Z16N110Z22N1451Z18N112Z22N1451Z18N112Z22N1447Z20N114Z20N1449Z20N114Z20N1447Z20N116Z20N1447Z20N116Z20N1445Z20N118Z20N1445Z20N118Z20N1441Z22N124Z14N1443Z22N124Z14N1437Z22N1581Z22N7626Z4N1599Z4N19697Z4N1599Z4N1599Z6N1597Z6N1597Z6N1597Z6N1595Z10N1593Z10N1593Z10N1593Z10N1593Z10N1593Z10N1595Z6N1597Z6N12928Z4N1599Z4N1595Z10N1593Z10N1591Z12N1591Z12N1597Z4N1599Z4N14345Z2N1601Z2N1597Z8N1595Z8N1597Z6N281380Z",
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 38123,
        "percentage_predicted_pixels": 0.015453075515896324,
    },
    {
        "label_name": "Brown Field",
        "label_index": 4,
        "score": 0.9460220578406374,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 674867,
        "percentage_predicted_pixels": 0.2735558774017366,
    },
    {
        "label_name": "Trees",
        "label_index": 5,
        "score": 0.9752337912343116,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 977533,
        "percentage_predicted_pixels": 0.39624088524724393,
    },
    {
        "label_name": "Structure",
        "label_index": 6,
        "score": 0.9666909771536597,
        "mask_shape": (1539, 1603),
        "num_predicted_pixels": 761316,
        "percentage_predicted_pixels": 0.3085977923946207,
    },
]


def test_od_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Endpoint: https://app.landing.ai/app/376/pr/11165/deployment?device=tiger-team-integration-tests
    endpoint_id = "db90b68d-cbfd-4a9c-8dc2-ebc4c3f6e5a4"
    predictor = Predictor(endpoint_id, api_key=_API_KEY)
    img = Image.open("tests/data/images/cereal1.jpeg")
    assert img is not None
    # Call LandingLens inference endpoint with Predictor.predict()
    preds = predictor.predict(
        img,
        metadata=InferenceMetadata(
            imageId="test-img-id-1",
            inspectionStationId="camera-station-1",
            locationId="factory-floor-1",
        ),
    )
    assert len(preds) == 3, "Result should not be empty or None"
    expected_scores = [0.9997851252555847, 0.9983770251274109, 0.9983124732971191]
    expected_bboxes = [
        (432, 1036, 652, 1203),
        (948, 1592, 1122, 1798),
        (1518, 1413, 1991, 1799),
    ]
    for i, pred in enumerate(preds):
        assert pred.label_name == "Screw"
        assert pred.label_index == 1
        assert pred.score == expected_scores[i]
        assert pred.bboxes == expected_bboxes[i]
    logging.info(preds)
    img_with_preds = overlay_predictions(predictions=preds, image=img)
    img_with_preds.save("tests/output/test_od.jpg")


def test_seg_predict(seg_mask_validator):
    expected_seg_prediction = {
        "label_name": "screw",
        "label_index": 1,
        "score": 0.9947679092064468,
        "encoded_mask": "2134598Z18N2030Z18N2030Z18N2022Z33N2015Z33N2015Z33N2010Z41N2007Z41N2004Z46N2002Z46N2002Z46N2000Z51N1997Z51N1992Z56N1992Z56N1992Z56N1986Z62N1986Z62N1979Z69N1979Z69N1979Z69N1974Z76N1972Z76N1967Z81N1967Z81N1967Z81N1964Z84N1964Z84N1964Z84N1961Z87N1961Z87N1961Z90N1958Z90N1958Z90N1956Z92N1956Z92N1953Z92N1956Z92N1956Z92N1954Z94N1954Z94N1946Z102N1946Z102N1946Z102N1910Z3N30Z105N1910Z3N30Z105N1892Z31N15Z110N1892Z31N15Z110N1892Z31N15Z110N1890Z41N2Z115N1890Z41N2Z115N1890Z41N2Z115N1890Z158N1890Z158N1887Z159N1889Z159N1889Z159N1889Z156N1892Z156N1892Z154N1894Z154N1894Z154N1894Z151N1897Z151N1897Z149N1899Z149N1899Z149N1899Z144N1904Z144N1904Z144N1904Z141N1907Z141N1907Z138N1910Z138N1910Z138N1910Z136N1912Z136N1912Z131N1917Z131N1917Z131N1917Z126N1922Z126N1922Z120N1928Z120N1928Z120N1928Z115N1933Z115N1936Z110N1938Z110N1938Z110N1938Z105N1943Z105N1943Z105N1943Z102N1946Z102N1946Z100N1948Z100N1948Z100N1950Z92N1956Z92N1959Z84N1964Z84N1964Z84N1966Z77N1971Z77N1974Z69N1979Z69N1979Z69N1979Z67N1981Z67N1984Z64N1984Z64N1984Z64N1984Z61N1987Z61N1987Z61N1989Z59N1989Z59N1992Z56N1992Z56N1992Z56N1992Z53N1995Z53N1997Z51N1997Z51N1997Z51N2002Z46N2002Z46N2005Z43N2005Z43N2005Z43N2010Z38N2010Z38N2010Z38N2013Z35N2013Z35N2015Z33N2015Z33N2015Z33N2020Z26N2022Z26N2032Z13N2035Z13N2035Z13N527703Z3N2045Z3N2038Z18N2030Z18N2030Z18N2022Z31N2017Z31N2012Z41N2007Z41N2007Z41N2004Z49N1999Z49N1999Z49N1994Z56N1992Z56N1990Z61N1987Z61N1987Z61N1984Z67N1981Z67N1978Z70N1978Z70N1978Z70N1973Z77N1971Z77N1953Z100N1948Z100N1948Z100N1943Z108N1940Z108N1935Z118N1930Z118N1930Z118N1923Z128N1920Z128N1920Z128N1914Z136N1912Z136N1907Z141N1907Z141N1907Z141N1902Z149N1899Z149N1894Z154N1894Z154N1894Z154N1891Z159N1889Z159N1884Z164N1884Z164N1884Z164N1879Z172N1876Z172N1876Z172N1871Z179N1869Z179N1861Z190N1858Z190N1858Z190N1853Z197N1851Z197N1846Z202N1846Z202N1846Z202N1844Z207N1841Z207N1836Z212N1836Z212N1836Z212N1833Z215N1833Z215N1828Z223N1825Z223N1825Z223N1820Z228N1820Z228N1820Z228N1817Z231N1817Z231N1812Z238N1810Z238N1810Z238N1800Z248N1800Z248N1797Z251N1797Z251N1797Z251N1792Z256N1792Z256N1787Z261N1787Z261N1787Z261N1782Z266N1782Z266N1779Z269N1779Z269N1779Z269N1774Z274N1774Z274N1774Z274N1772Z276N1772Z276N1769Z279N1769Z279N1769Z279N1761Z287N1761Z287N1751Z295N1753Z295N1753Z295N1735Z313N1735Z313N1725Z320N1728Z320N1728Z320N1720Z325N1723Z325N1723Z325N1721Z327N1721Z327N1718Z328N1720Z328N1720Z328N1718Z327N1721Z327N1713Z333N1715Z333N1715Z333N1674Z374N1674Z374N1666Z379N1669Z379N1669Z379N1659Z389N1659Z389N1649Z397N1651Z397N1651Z397N1646Z399N1649Z399N1649Z399N1646Z402N1646Z402N1644Z402N1646Z402N1646Z402N1646Z399N1649Z399N1651Z394N1654Z394N1654Z394N1657Z386N1662Z386N1662Z381N1667Z381N1667Z381N1667Z376N1672Z376N1674Z369N1679Z369N1679Z369N1679Z364N1684Z364N1684Z364N1684Z361N1687Z361N1687Z356N1692Z356N1692Z356N1692Z354N1694Z354N1692Z353N1695Z353N1695Z353N1695Z348N1700Z348N1700Z343N1705Z343N1705Z343N1152Z20N533Z338N1157Z20N533Z338N1157Z20N533Z338N1154Z28N525Z338N1157Z28N525Z338N1155Z35N520Z333N1160Z35N520Z333N1160Z35N520Z333N1160Z38N517Z330N1163Z38N517Z330N1160Z46N515Z309N1178Z46N515Z309N1178Z46N515Z309N1178Z54N507Z302N1185Z54N507Z302N1183Z66N499Z295N1188Z66N499Z295N1188Z66N499Z295N1188Z74N494Z287N1193Z74N494Z287N1190Z84N489Z279N1196Z84N489Z279N1196Z84N489Z279N1196Z90N486Z269N1203Z90N486Z269N1203Z90N486Z269N1200Z95N489Z259N1205Z95N489Z259N1205Z95N494Z246N1213Z95N494Z246N1213Z95N494Z246N1213Z95N497Z235N1221Z95N497Z235N1221Z95N502Z223N1228Z95N502Z223N1228Z95N502Z223N1228Z95N502Z217N1234Z95N502Z217N1234Z95N504Z210N1239Z95N504Z210N1239Z95N504Z210N1239Z95N507Z205N1241Z95N507Z205N1241Z95N507Z200N1246Z95N507Z200N1246Z95N507Z200N1249Z92N510Z194N1252Z92N510Z194N1252Z92N510Z194N1252Z95N507Z192N1254Z95N507Z192N1259Z90N507Z184N1267Z90N507Z184N1267Z90N507Z184N1270Z89N507Z177N1275Z89N507Z177N1277Z87N507Z172N1282Z87N507Z172N1282Z87N507Z172N1285Z87N504Z167N1290Z87N504Z167N1295Z82N504Z161N1301Z82N504Z161N1301Z82N504Z161N1303Z80N504Z154N1310Z80N504Z154N1310Z80N504Z154N1313Z77N504Z146N1321Z77N504Z146N1321Z77N504Z143N1324Z77N504Z143N1324Z77N504Z143N1321Z77N507Z141N1323Z77N507Z141N1323Z77N510Z135N1326Z77N510Z135N1326Z77N510Z135N1326Z75N512Z133N1328Z75N512Z133N1326Z74N515Z133N1326Z74N515Z133N1326Z74N515Z133N1326Z74N515Z130N1329Z74N515Z130N1329Z74N515Z130N1329Z74N515Z130N1329Z74N515Z130N1329Z72N517Z130N1329Z72N517Z130N1329Z72N517Z130N1326Z75N517Z128N1328Z75N517Z128N1328Z75N519Z126N1328Z75N519Z126N1328Z75N519Z126N1326Z77N522Z123N1326Z77N522Z123N1326Z77N524Z121N1326Z77N524Z121N1326Z77N524Z121N1323Z80N524Z121N1323Z80N524Z121N1321Z82N527Z118N1321Z82N527Z118N1321Z82N527Z118N1321Z79N530Z118N1321Z79N530Z118N1321Z79N530Z118N1321Z79N530Z118N1321Z79N530Z118N1318Z82N532Z116N1318Z82N532Z116N1318Z82N532Z116N1318Z82N532Z113N1321Z82N532Z113N1321Z79N538Z110N1321Z79N538Z110N1321Z79N538Z110N1321Z77N540Z108N1323Z77N540Z108N1323Z77N543Z102N1326Z77N543Z102N1326Z77N543Z102N1326Z72N550Z97N1329Z72N550Z97N1329Z69N556Z94N1329Z69N556Z94N1329Z69N556Z94N1329Z67N560Z90N1331Z67N560Z90N1331Z67N560Z90N1331Z67N565Z82N1334Z67N565Z82N1334Z64N574Z76N1334Z64N574Z76N1334Z64N574Z76N1334Z64N579Z69N1336Z64N579Z69N1336Z64N581Z62N1341Z64N581Z62N1341Z64N581Z62N1341Z64N584Z56N1344Z64N584Z56N1344Z62N588Z49N1349Z62N588Z49N1349Z62N588Z49N1349Z62N591Z43N1352Z62N591Z43N1352Z62N593Z39N1354Z62N593Z39N1354Z62N593Z39N1357Z59N596Z33N1360Z59N596Z33N1360Z59N596Z33N1360Z56N602Z25N1365Z56N602Z25N1367Z51N1997Z51N1997Z51N2000Z46N2002Z46N2007Z36N2012Z36N2012Z36N2017Z23N2025Z23N2028Z15N2033Z15N2033Z15N2035Z10N2038Z10N2041Z5N2043Z5N2043Z5N519175Z",
        "num_predicted_pixels": 93640,
        "percentage_predicted_pixels": 0.022325515747070312,
        "mask_shape": (2048, 2048),
    }
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26113016987660/deployment?device=tiger-team-integration-tests
    endpoint_id = "72fdc6c2-20f1-4f5e-8df4-62387acec6e4"
    predictor = Predictor(endpoint_id, api_key=_API_KEY)
    img = Image.open("tests/data/images/cereal1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    seg_mask_validator(preds[0], expected_seg_prediction)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_seg.jpg")


def test_vp_predict(seg_mask_validator):
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26098103179275/deployment?device=tiger-example
    endpoint_id = "63035608-9d24-4342-8042-e4b08e084fde"
    predictor = Predictor(endpoint_id, api_key=_API_KEY)
    img = np.asarray(Image.open("tests/data/images/farm-coverage.jpg"))
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 4, "Result should not be empty or None"
    for actual, expected in zip(preds, _EXPECTED_VP_PREDS):
        seg_mask_validator(actual, expected)
    color_map = {
        "Trees": "green",
        "Structure": "#FFFF00",  # yellow
        "Brown Field": "red",
        "Green Field": "blue",
    }
    options = {"color_map": color_map}
    img_with_masks = overlay_predictions(preds, img, options).resize((512, 512))
    img_with_masks.save("tests/output/test_vp.png")
    expected = Image.open("tests/data/images/expected_vp_masks.png")
    diff = ImageChops.difference(img_with_masks, expected)
    assert diff.getbbox() is None, "Expected and actual images should be the same"


def test_class_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    # Project: https://app.landing.ai/app/376/pr/26119078438913/deployment?device=tiger-team-integration-tests
    endpoint_id = "8fc1bc53-c5c1-4154-8cc1-a08f2e17ba43"
    predictor = Predictor(endpoint_id, api_key=_API_KEY)
    img = Image.open("tests/data/images/wildfire1.jpeg")
    assert img is not None
    preds = predictor.predict(img)
    assert len(preds) == 1, "Result should not be empty or None"
    assert preds[0].label_name == "HasFire"
    assert preds[0].label_index == 0
    assert preds[0].score == 0.995685338973999
    logging.info(preds)
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_class.jpg")


# TODO: re-enable below test after OCR endpoint is deployed to prod
@pytest.mark.skip(reason="OCR endpoint is not deployed to prod yet")
def test_ocr_predict():
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    predictor = OcrPredictor(
        # TODO: replace with a prod key after the OCR endpoint is deployed to prod
        api_key="",
    )
    img = Image.open("tests/data/images/ocr_test.png")
    assert img is not None
    # Test multi line
    preds = predictor.predict(img, mode="multi-text")
    logging.info(preds)
    expected_texts = [
        "公司名称",
        "业务方向",
        "Anysphere",
        "AI工具",
        "Atomic Semi",
        "芯片",
        "Cursor",
        "代码编辑",
        "Diagram",
        "设计",
        "Harvey",
        "AI法律顾问",
        "Kick",
        "会计软件",
        "Milo",
        "家长虚拟助理",
        "qqbot.dev",
        "开发者工具",
        "EdgeDB",
        "开源数据库",
        "Mem Labs",
        "笔记应用",
        "Speak",
        "英语学习",
        "Descript",
        "音视频编辑",
        "量子位",
    ]
    preds = sorted(preds, key=lambda x: x.text)
    expected_texts = sorted(expected_texts)
    assert len(preds) == len(expected_texts)
    for pred, expected in zip(preds, expected_texts):
        assert pred.text == expected

    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_ocr_multiline.jpg")
    # Test single line
    preds = predictor.predict(
        img,
        mode="single-text",
        regions_of_interest=[
            [[99, 19], [366, 19], [366, 75], [99, 75]],
            [[599, 842], [814, 845], [814, 894], [599, 892]],
        ],
    )
    logging.info(preds)
    expected = [
        {
            "text": "公司名称",
            "location": [(99, 19), (366, 19), (366, 75), (99, 75)],
            "score": 0.8279303908348083,
        },
        {
            "text": "英语学习",
            "location": [(599, 842), (814, 845), (814, 894), (599, 892)],
            "score": 0.939440906047821,
        },
    ]
    for pred, expected in zip(preds, expected):
        assert pred.text == expected["text"]
        assert pred.location == expected["location"]
        assert pred.score == expected["score"]
    img_with_masks = overlay_predictions(preds, img)
    img_with_masks.save("tests/output/test_ocr_singleline.jpg")
