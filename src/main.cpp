#include "coverage_planner.h"

#define DENSE_PATH 1


/**
 * 对图像进行处理，膨胀边界和障碍物
*/
cv::Mat Contour_treatment(cv::Mat &img)
{
    // cv::Mat img = cv::imread("../data/basement.png");
    
    cv::imshow("原图像", img);
    cv::waitKey(500);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // 把图像修改为二进制灰度图

    cv::Mat img_ = gray.clone(); // 储存二进制灰度图

    cv::threshold(img_, img_, 250, 255, 0); // 去除灰度图上的噪点
    std::cout<<"去掉噪，例如过滤很小或很大像素值的图像点。"<<std::endl;

    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10,10), cv::Point(-1,-1)); // 返回元素卷积核
    cv::morphologyEx(img_, img_, cv::MORPH_ERODE, erode_kernel); // 对卷积图像做边界的膨胀
    cv::imshow("边界膨胀", img_);
    cv::waitKey();
    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5), cv::Point(-1,-1)); // 返回元素卷积核
    cv::morphologyEx(img_, img_, cv::MORPH_OPEN, open_kernel); // 对卷积图像做开运算，做腐蚀处理
    std::cout<<"返回一个结构元素（卷积核）"<<std::endl;
    cv::imshow("腐蚀处理", img_);
    cv::waitKey();
    // cv::waitKey(500);
    return img_;
}

/**
 * 对图像进行处理，圈出边界和各个障碍物，得到数组数据
*/
void Circle_boundary(cv::Mat &img_, cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{
    /*** 查找图像的轮廓，输出为输出的轮廓集合，每个轮廓由一系列点组成。输出的轮廓层次结构，用于表示轮廓之间的父子关系。 ****/
    std::vector<std::vector<cv::Point>> cnts; // 储存多个多边形的轮廓外形数据
    std::vector<cv::Vec4i> hierarchy; // 配合多边形轮廓数组，保存多个多边形的内外父子关系 index: next, prev, first_child, parent
    cv::findContours(img_, cnts, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE); // 找到轮廓数据和父子关系并储存在变量中
    std::cout<<"在二值图像中检测轮廓。它可以将图像中的连续区域(通常是物体)提取出来，形成一个轮廓集合"<<std::endl;

    /***对多边形数据转存排序******************/
    std::vector<int> cnt_indices(cnts.size()); // 转存多边形数据
    std::iota(cnt_indices.begin(), cnt_indices.end(), 0); // 为cnt_indices批量赋值为0
    std::sort(cnt_indices.begin(), cnt_indices.end(), [&cnts](int lhs, int rhs){return cv::contourArea(cnts[lhs]) > cv::contourArea(cnts[rhs]);}); // 排序 
    int ext_cnt_idx = cnt_indices.front(); // 返回的是第一个元素的引用

    /***展示外部边界***********************/
    cv::Mat cnt_canvas = img.clone(); // 复制原图像
    cv::drawContours(cnt_canvas, cnts, ext_cnt_idx, cv::Scalar(0,0,255)); // 在原图像上绘制轮廓
    cv::imshow("绘制外部边界", cnt_canvas);
    cv::waitKey();

    /***展示内部障碍物边界***********************/
    // std::vector<std::vector<cv::Point>> contours; 转而使用外部输入
    contours.emplace_back(cnts[ext_cnt_idx]); // 把处理好的cnts轮廓数据添加到新的轮廓变量中
    // 找到所有的轮廓的障碍物
    for(int i = 0; i < hierarchy.size(); i++){ // 找到最外围的父轮廓，里面的子轮廓就是障碍物
        if(hierarchy[i][3]==ext_cnt_idx){ //父轮廓的索引等于外部轮廓的索引
            contours.emplace_back(cnts[i]);
            cv::drawContours(cnt_canvas, cnts, i, cv::Scalar(255,0,0)); // 把障碍的轮廓绘制到图像中展示
        }
    }
    cv::imshow("找到所有的障碍物轮廓", cnt_canvas);
    std::cout<<"找到所有的障碍物轮廓"<<std::endl;
    cv::waitKey();
    // cv::waitKey(1000);

    //////////////////////////////仅做展示，展示障碍物边界，不参与逻辑/////////////////////////////////////////
    cv::Mat cnt_img = cv::Mat(img.rows, img.cols, CV_8UC3);
    cnt_img.setTo(255);
    for(int i = 0; i < contours.size(); i++){
        cv::drawContours(cnt_img, contours, i, cv::Scalar(0,0,0));
    }
    cv::imshow("展示只保留轮廓的图形", cnt_img);
    std::cout<<"处理后只保留轮廓的图形"<<std::endl;
    cv::waitKey();
    // cv::waitKey(1000);
    ////////////////////////////仅做展示，不参与逻辑 end///////////////////////////////////////////
}

/**
 * 对边界和障碍物的外形抽象成简单的多边形
*/
void Polygon_approximation(cv::Mat &img_, 
                            cv::Mat &img, 
                            std::vector<std::vector<cv::Point>> &contours, 
                            std::vector<std::vector<cv::Point>> &polys)
{
    /***处理内部边界，对每个内部图形的边界做转换简单多边形的逼近操作，把边界转换成简单多边形边界***********************/
    cv::Mat poly_canvas = img.clone();
    std::vector<cv::Point> poly; // 储存外部简单多边形边界
    // std::vector<std::vector<cv::Point>> polys; // 储存内部障碍物轮廓多边形边界
    for(auto& contour : contours){
        cv::approxPolyDP(contour, poly, 3, true); // 对每个内部轮廓都做处理，变为更逼近的多边形外形
        polys.emplace_back(poly);
        poly.clear();
    }
    for(int i = 0; i < polys.size(); i++){
        cv::drawContours(poly_canvas, polys, i, cv::Scalar(255,0,255));
        cv::imshow("多边形逼近", poly_canvas);
        std::cout<<"将复杂的多边形转换成简单的多边形"<<std::endl;
        cv::waitKey(500);
    }
    /////////////仅做展示，展示内部多边形边界//////////////////////////////////////////////////////////
    cv::Mat poly_img = cv::Mat(img.rows, img.cols, CV_8UC3);
    poly_img.setTo(255);
    for(int i = 0; i < polys.size(); i++){
        cv::drawContours(poly_img, polys, i, cv::Scalar(0,0,0));
    }
    cv::imshow("only polygons 只保留多边形的图形", poly_img);
    std::cout<<"只保留多边形的图形"<<std::endl;
    // cv::waitKey();
    cv::waitKey(1000);

    /***找到地图外部的边界线，储存在变量中***********************/
    std::vector<int> line_deg_histogram(180); // 用于保存外部边界的角度数据
    double line_len; // 直线长度
    double line_deg; // DEG角度 直线角度
    int line_deg_idx; // 索引、指示
    cv::Mat line_canvas = img.clone();
    auto ext_poly = polys.front(); // 储存内部障碍物轮廓多边形边界

    ext_poly.emplace_back(ext_poly.front()); // 数组赋值
    for(int i = 1; i < ext_poly.size(); i++){
        line_len = std::sqrt(std::pow((ext_poly[i].x-ext_poly[i-1].x),2)+std::pow((ext_poly[i].y-ext_poly[i-1].y),2)); // 计算边界长度
        // y-axis towards up, x-axis towards right, theta is from x-axis to y-axis
        line_deg = std::round(atan2(-(ext_poly[i].y-ext_poly[i-1].y),(ext_poly[i].x)-ext_poly[i-1].x)/M_PI*180.0); // atan2: (-180, 180]
        line_deg_idx = (int(line_deg) + 180) % 180; // [0, 180)
        line_deg_histogram[line_deg_idx] += int(line_len);

       std::cout<<"deg: "<<line_deg_idx<<std::endl;
       cv::line(line_canvas, ext_poly[i], ext_poly[i-1], cv::Scalar(255,255,0));
       cv::imshow("生成最外围地图线",line_canvas);
       cv::waitKey();
    }
    cv::imshow("生成最外围地图线",line_canvas);
    // cv::waitKey();
    cv::waitKey(1000);
    

    auto it = std::max_element(line_deg_histogram.begin(), line_deg_histogram.end()); // 求数组中的最大值
    int main_deg = (it-line_deg_histogram.begin());
    std::cout<<"main deg: " << main_deg<<std::endl;
}

void cell_decomposition(cv::Mat &img_, 
                        cv::Mat &img, 
                        std::vector<std::vector<cv::Point>> &polys, // 储存内部障碍物轮廓多边形边界
                        std::vector<Polygon_2> &bcd_cells // 用于储存被分解出来的细胞部分
                        )
{
    // construct polygon with holes
    std::cout<<"用细胞结构建多边形"<<std::endl;

    std::vector<cv::Point> outer_poly = polys.front();
    polys.erase(polys.begin());
    std::vector<std::vector<cv::Point>> inner_polys = polys;

    Polygon_2 outer_polygon;
    for(const auto& point : outer_poly){
        outer_polygon.push_back(Point_2(point.x, point.y));
    }

    int num_holes = inner_polys.size();
    std::vector<Polygon_2> holes(num_holes);
    for(int i = 0; i < inner_polys.size(); i++){
        for(const auto& point : inner_polys[i]){
            holes[i].push_back(Point_2(point.x, point.y));
        }
    }
    // 定义用于处理的数据
    PolygonWithHoles pwh(outer_polygon, holes.begin(), holes.end());

    ///////////////////////////////////////////////////////////////////////
    // cell decomposition

    // std::vector<Polygon_2> bcd_cells; // 用于储存被分解出来的细胞部分
    std::cout<<"细胞分解化分解中..........."<<std::endl;
//    polygon_coverage_planning::computeBestTCDFromPolygonWithHoles(pwh, &bcd_cells);
    polygon_coverage_planning::computeBestBCDFromPolygonWithHoles(pwh, &bcd_cells);
    std::cout<<"polygon_coverage_planning end ! suddessful"<<std::endl;

    ////////////////////////////下面仅仅演示细胞路径/////////////////////////////////////////////
    // test decomposition
    std::vector<std::vector<cv::Point>> bcd_polys;
    std::vector<cv::Point> bcd_poly;

    for(const auto& cell:bcd_cells){
        for(int i = 0; i < cell.size(); i++){
            bcd_poly.emplace_back(cv::Point(CGAL::to_double(cell[i].x()), CGAL::to_double(cell[i].y())));
        }
        bcd_polys.emplace_back(bcd_poly);
        bcd_poly.clear();
    }
    cv::Mat poly_img = cv::Mat(img.rows, img.cols, CV_8UC3);
    poly_img.setTo(255);
    for(int i = 0; i < bcd_polys.size(); i++){
        cv::drawContours(poly_img, bcd_polys, i, cv::Scalar(255,0,255));
        cv::imshow("bcd 展示细胞化分解图片的过程", poly_img);
        cv::waitKey(30);
    }
    cv::imshow("bcd 展示细胞化分解图片的最终结果", poly_img);
    cv::waitKey();
    ///////end test decomposition//////////////////////////////////////////////////////////////////////////////.

    ////////////////////////////下面仅仅演示构造邻图/////////////////////////////////////////////
    // construct adjacent graph
//    std::map<size_t, std::set<size_t>> cell_graph;
//    bool succeed = calculateDecompositionAdjacency(bcd_cells, &cell_graph);
//
//    if(succeed){
//        std::cout<<"cell graph was constructed"<<std::endl;
//    }else{
//        std::cout<<"cell graph wasn't constructed"<<std::endl;
//    }
//
//    for(auto cell_it=cell_graph.begin(); cell_it!=cell_graph.end(); cell_it++){
//        std::cout<<"cell "<<cell_it->first<<" 's neighbors: "<<std::endl;
//        for(auto it=cell_it->second.begin(); it!=cell_it->second.end(); it++){
//            std::cout<<*it<<" ";
//        }
//        std::cout<<std::endl<<std::endl;
//    }
}

/**计算细胞之间的连接路径*/
void calculate_cell_path(cv::Mat &img_,  
                         cv::Mat &img, 
                         std::vector<Polygon_2> &bcd_cells, // 用于储存被分解出来的细胞部分
                         std::vector<CellNode> &cell_graph, // 对外输出细胞图
                         std::vector<std::vector<Point_2>> &cells_sweeps,
                         std::vector<std::map<int, std::list<Point_2 >>> &cell_intersections,
                         Point_2 &start,
                         std::deque<int> &cell_idx_path
                        )
{
    cell_graph = calculateDecompositionAdjacency(bcd_cells); // 计算得到细胞图
    for(auto& cell:cell_graph){
        std::cout<<"cell "<<cell.cellIndex<<" 's neighbors: ";
        for(auto& neighbor:cell.neighbor_indices){
            std::cout<<neighbor<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<< "计算分解邻接......." << std::endl;
    
    // Point_2 start;
    start = getStartingPoint(img); // 从窗口上获取一个路径启始位置
    std::cout<< "getStartingPoint" << std::endl;
    int starting_cell_idx = getCellIndexOfPoint(bcd_cells, start); // 定义路径起始位置
    std::cout<< "获取行进路径: " << starting_cell_idx << std::endl;
    // std::deque<int> cell_idx_path;
    cell_idx_path = getTravellingPath(cell_graph, starting_cell_idx); // 定义 细胞路径索引,每个路径点的排序
    std::cout<<"path length: "<<cell_idx_path.size()<<std::endl;
    std::cout<<"start";
    for(auto& cell_idx:cell_idx_path){
        std::cout<<"->"<<cell_idx;
    }
    std::cout<< "计算分解邻接完成" << std::endl;
    // cv::waitKey();

    int sweep_step = 10; // 分割间隙参数

    // std::vector<std::vector<Point_2>> cells_sweeps;

    for (size_t i = 0; i < bcd_cells.size(); ++i) {
        // Compute all cluster sweeps. 计算所有细胞内的路径规划
        std::vector<Point_2> cell_sweep;
        Direction_2 best_dir;
        polygon_coverage_planning::findBestSweepDir(bcd_cells[i], &best_dir); // 查找最佳扫描目录
        polygon_coverage_planning::visibility_graph::VisibilityGraph vis_graph(bcd_cells[i]);

        bool counter_clockwise = true;
        polygon_coverage_planning::computeSweep(bcd_cells[i], vis_graph, sweep_step, best_dir, counter_clockwise, &cell_sweep);
        cells_sweeps.emplace_back(cell_sweep); // 删除最后一个元素
    }

    ////////下面部分仅仅用于展示//////////////////////////////////////////////////////////
    // 用于分别展示每个细胞框架内的路径规划，下面部分仅仅用于展示
    cv::Point prev, curr;
    cv::Mat poly_img_ = img.clone();
    for(size_t i = 0; i < cells_sweeps.size(); ++i){
        for(size_t j = 1; j < cells_sweeps[i].size(); ++j){
            prev = cv::Point(CGAL::to_double(cells_sweeps[i][j-1].x()),CGAL::to_double(cells_sweeps[i][j-1].y()));
            curr = cv::Point(CGAL::to_double(cells_sweeps[i][j].x()),CGAL::to_double(cells_sweeps[i][j].y()));
            cv::line(poly_img_, prev, curr, cv::Scalar(0, 0, 255));
            cv::namedWindow("way",cv::WINDOW_NORMAL);
            cv::imshow("对每个细胞内部进行路径规划", poly_img_);
            // cv::waitKey(0);
            cv::waitKey(30);
        }
    }
    // cv::waitKey();
    //////////展示结束//////////////////////////////////////////////////////

    ////////////////仅展示代码过程//////////////////////
    // for(int i = 0; i < cell_graph.size(); i++){
    //     cv::drawContours(poly_img, bcd_polys, cell_graph[i].cellIndex, cv::Scalar(0, 255, 255));
    //     cv::namedWindow("path cell", cv::WINDOW_NORMAL);
    //     cv::imshow("path cell", poly_img);
    //     cv::waitKey(500);
    //     cv::drawContours(poly_img, bcd_polys, cell_graph[i].cellIndex, cv::Scalar(0, 0, 255));
    // }
    // cv::waitKey();
    // std::vector<std::map<int, std::list<Point_2 >>> cell_intersections;
    cell_intersections = calculateCellIntersections(bcd_cells, cell_graph);

//    for(size_t i = 0; i < cell_intersections.size(); ++i){
//        for(auto j = cell_intersections[i].begin(); j != cell_intersections[i].end(); ++j){
//            std::cout<<"cell "<<i<<" intersect with "<<"cell "<<j->first<<": ";
//            for(auto k = j->second.begin(); k != j->second.end(); ++k){
//                std::cout<<"("<<CGAL::to_double(k->x())<<", "<<CGAL::to_double(k->y())<<")"<<" ";
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl<<std::endl;
//    }

}

int main(){
    cv::Mat img = cv::imread("../data/test1.png");

    cv::Mat img_ = Contour_treatment(img); // 对图像进行处理，膨胀边界和障碍物

    std::vector<std::vector<cv::Point>> contours; // 保存障碍物位置
    Circle_boundary(img_, img, contours); // 对图像进行处理，圈出边界和各个障碍物，得到数组数据
    
    std::vector<std::vector<cv::Point>> polys; // 储存内部障碍物轮廓多边形边界
    Polygon_approximation(img_, img, contours, polys); // 对边界和障碍物的外形抽象成简单的多边形
    
    std::vector<Polygon_2> bcd_cells; // 用于储存被分解出来的细胞部分
    cell_decomposition(img_, img, polys, bcd_cells); // 把图形分割成一个个细胞块

    ///////////////////////////////////////////////////////////////////////
    std::vector<CellNode> cell_graph; // 对外输出细胞图
    std::vector<std::vector<Point_2>> cells_sweeps;
    std::vector<std::map<int, std::list<Point_2 >>> cell_intersections;
    Point_2 start;
    std::deque<int> cell_idx_path;
    calculate_cell_path(img_, img, bcd_cells, cell_graph, cells_sweeps, cell_intersections, start, cell_idx_path);
    ////////////////////////////////////////////////////

    std::vector<Point_2> way_points;

#if DENSE_PATH /// 使用稠密路径
    Point_2 point = start;
    std::list<Point_2> next_candidates;
    Point_2 next_point;
    std::vector<Point_2> shortest_path;

    // 执行反向下一个扫描
    if(doReverseNextSweep(start, cells_sweeps[cell_idx_path.front()])){ 
        // 找到下一个最短的路径
        shortest_path = getShortestPath(bcd_cells[cell_idx_path.front()], start, cells_sweeps[cell_idx_path.front()].back());
        // 将路径点添加到路径中
        way_points.insert(way_points.end(), shortest_path.begin(), std::prev(shortest_path.end()));
    } else{
        shortest_path = getShortestPath(bcd_cells[cell_idx_path.front()], start, cells_sweeps[cell_idx_path.front()].front());
        way_points.insert(way_points.end(), shortest_path.begin(), std::prev(shortest_path.end()));
    }

    point = way_points.back(); // 返回的的是最后一个元素的引用。

    for(size_t i = 0; i < cell_idx_path.size(); ++i){
        // has been cleaned?
        if(!cell_graph[cell_idx_path[i]].isCleaned){ // 判断是否被清理
            // need to reverse?
            if(doReverseNextSweep(point, cells_sweeps[cell_idx_path[i]])){
                way_points.insert(way_points.end(), cells_sweeps[cell_idx_path[i]].rbegin(), cells_sweeps[cell_idx_path[i]].rend());
            }else{
                way_points.insert(way_points.end(), cells_sweeps[cell_idx_path[i]].begin(), cells_sweeps[cell_idx_path[i]].end());
            }
            // now cleaned
            cell_graph[cell_idx_path[i]].isCleaned = true;
            // update current point
            point = way_points.back();
            // find shortest path to next cell
            if((i+1)<cell_idx_path.size()){
                next_candidates = cell_intersections[cell_idx_path[i]][cell_idx_path[i+1]];
                if(doReverseNextSweep(point, cells_sweeps[cell_idx_path[i+1]])){
                    next_point = findNextGoal(point, cells_sweeps[cell_idx_path[i+1]].back(), next_candidates);
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
                    way_points.insert(way_points.end(), std::next(shortest_path.begin()), std::prev(shortest_path.end()));
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i+1]], next_point, cells_sweeps[cell_idx_path[i+1]].back());
                }else{
                    next_point = findNextGoal(point, cells_sweeps[cell_idx_path[i+1]].front(), next_candidates);
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
                    way_points.insert(way_points.end(), std::next(shortest_path.begin()), std::prev(shortest_path.end()));
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i+1]], next_point, cells_sweeps[cell_idx_path[i+1]].front());
                }
                way_points.insert(way_points.end(), shortest_path.begin(), std::prev(shortest_path.end()));
                point = way_points.back();
            }
        }else{
            shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]],
                                            cells_sweeps[cell_idx_path[i]].front(),
                                            cells_sweeps[cell_idx_path[i]].back());
            if(doReverseNextSweep(point, cells_sweeps[cell_idx_path[i]])){
                way_points.insert(way_points.end(), shortest_path.rbegin(), shortest_path.rend());
            }else{
                way_points.insert(way_points.end(), shortest_path.begin(), shortest_path.end());
            }
            point = way_points.back();

            if((i+1)<cell_idx_path.size()){
                next_candidates = cell_intersections[cell_idx_path[i]][cell_idx_path[i+1]];
                if(doReverseNextSweep(point, cells_sweeps[cell_idx_path[i+1]])){
                    next_point = findNextGoal(point, cells_sweeps[cell_idx_path[i+1]].back(), next_candidates);
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
                    way_points.insert(way_points.end(), std::next(shortest_path.begin()), std::prev(shortest_path.end()));
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i+1]], next_point, cells_sweeps[cell_idx_path[i+1]].back());
                }else{
                    next_point = findNextGoal(point, cells_sweeps[cell_idx_path[i+1]].front(), next_candidates);
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
                    way_points.insert(way_points.end(), std::next(shortest_path.begin()), std::prev(shortest_path.end()));
                    shortest_path = getShortestPath(bcd_cells[cell_idx_path[i+1]], next_point, cells_sweeps[cell_idx_path[i+1]].front());
                }
                way_points.insert(way_points.end(), shortest_path.begin(), std::prev(shortest_path.end()));
                point = way_points.back();
            }
        }
    }

    cv::Point p1, p2;
    // cv::namedWindow("cover",cv::WINDOW_NORMAL);
    // cv::imshow("cover", img);
    // cv::waitKey();
    for(size_t i = 1; i < way_points.size(); ++i){
        p1 = cv::Point(CGAL::to_double(way_points[i-1].x()),CGAL::to_double(way_points[i-1].y()));
        p2 = cv::Point(CGAL::to_double(way_points[i].x()),CGAL::to_double(way_points[i].y()));
        cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
        cv::namedWindow("cover",cv::WINDOW_NORMAL);
        cv::imshow("展示路径规划过程", img);
        cv::waitKey(50);
        cv::line(img, p1, p2, cv::Scalar(200, 200, 200));
        std::cout<<"P1("<<p1.x<<","<<p1.y<<")"<<" P1("<<p1.x<<","<<p1.y<<")"<<std::endl;
    }
    cv::imshow("最终路径", img);
    cv::waitKey(2000);
    cv::waitKey();
#else

    cv::Point p1, p2;
    cv::Mat temp_img;
    cv::namedWindow("cover",cv::WINDOW_NORMAL);
    cv::imshow("cover", img);
    cv::waitKey();

    Point_2 point = start;
    way_points.emplace_back(point);

    for(auto& idx : cell_idx_path){
        if(!cell_graph[idx].isCleaned){
            if(doReverseNextSweep(point, cells_sweeps[idx])){
                way_points.insert(way_points.end(), cells_sweeps[idx].rbegin(), cells_sweeps[idx].rend());

                temp_img = img.clone();
                cv::line(img,
                        cv::Point(CGAL::to_double(point.x()),CGAL::to_double(point.y())),
                        cv::Point(CGAL::to_double(cells_sweeps[idx].back().x()),CGAL::to_double(cells_sweeps[idx].back().y())),
                        cv::Scalar(255, 0, 0),
                        1);
                cv::namedWindow("cover",cv::WINDOW_NORMAL);
                cv::imshow("cover", img);
//                cv::waitKey(500);
                img = temp_img.clone();

                for(size_t i = (cells_sweeps[idx].size()-1); i > 0; --i){
                    p1 = cv::Point(CGAL::to_double(cells_sweeps[idx][i].x()),CGAL::to_double(cells_sweeps[idx][i].y()));
                    p2 = cv::Point(CGAL::to_double(cells_sweeps[idx][i-1].x()),CGAL::to_double(cells_sweeps[idx][i-1].y()));
                    cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
                    cv::namedWindow("cover",cv::WINDOW_NORMAL);
                    cv::imshow("cover", img);
//                    cv::waitKey(50);
                    cv::line(img, p1, p2, cv::Scalar(200, 200, 200));
                }

            }else{
                way_points.insert(way_points.end(), cells_sweeps[idx].begin(), cells_sweeps[idx].end());

                temp_img = img.clone();
                cv::line(img,
                         cv::Point(CGAL::to_double(point.x()),CGAL::to_double(point.y())),
                         cv::Point(CGAL::to_double(cells_sweeps[idx].front().x()),CGAL::to_double(cells_sweeps[idx].front().y())),
                         cv::Scalar(255, 0, 0),
                         1);
                cv::namedWindow("cover",cv::WINDOW_NORMAL);
                cv::imshow("cover", img);
//                cv::waitKey(500);
                img = temp_img.clone();

                for(size_t i = 1; i < cells_sweeps[idx].size(); ++i){
                    p1 = cv::Point(CGAL::to_double(cells_sweeps[idx][i-1].x()),CGAL::to_double(cells_sweeps[idx][i-1].y()));
                    p2 = cv::Point(CGAL::to_double(cells_sweeps[idx][i].x()),CGAL::to_double(cells_sweeps[idx][i].y()));
                    cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
                    cv::namedWindow("cover",cv::WINDOW_NORMAL);
                    cv::imshow("cover", img);
//                    cv::waitKey(50);
                    cv::line(img, p1, p2, cv::Scalar(200, 200, 200));
                }
            }

            cell_graph[idx].isCleaned = true;
            point = way_points.back();
        }
    }

    cv::waitKey();

#endif

    return 0;
}
